import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from evaluate_td3_gnn import (
    ALGORITHM_CHOICES,
    create_policy,
    load_checkpoint_kwargs,
    load_policy_checkpoint,
    make_env,
    normalise_algorithm_label,
    normalise_checkpoint_prefix,
    normalise_step_result,
    reset_env_state,
    validate_checkpoint_identity,
)
from scripts.transformer_constraint_projection_diagnostics import (
    PROJECTION_DIAGNOSTIC_TOLERANCE,
    PROJECTION_EPISODE_SUMMARY_COLUMNS,
    PROJECTION_SEED_SUMMARY_COLUMNS,
    PROJECTION_TRANSFORMER_STEP_COLUMNS,
    summarise_projection_episode,
    summarise_projection_seed,
    transformer_step_rows_from_projection_details,
    write_csv_rows,
)
from utils.ev2gym_training_utils import resolve_device, str2bool
from utils.transformer_feasibility_projection import project_transformer_feasible_actions


SCALE_CHOICES = ("25cp", "100cp", "500cp", "1000cp")


@dataclass(frozen=True)
class ProjectionActionDiagnostics:
    raw_action: torch.Tensor
    safe_action: torch.Tensor
    mapped_action: np.ndarray
    transformer_rows: list[dict[str, object]]
    corrected_ev_decisions: int
    active_ev_decisions: int


def _active_ev_count_by_transformer(state) -> dict[object, int]:
    active_ev_indexes = torch.as_tensor(state.ev_indexes, dtype=torch.long).reshape(-1)
    if active_ev_indexes.numel() == 0:
        return {}
    ev_features = torch.as_tensor(state.ev_features, dtype=torch.float32)
    counts: dict[object, int] = {}
    graph_node_start = 0
    for graph_index, sample_node_length in enumerate(state.sample_node_length):
        graph_node_end = graph_node_start + int(sample_node_length)
        graph_mask = (active_ev_indexes >= graph_node_start) & (active_ev_indexes < graph_node_end)
        graph_ev_positions = torch.nonzero(graph_mask, as_tuple=False).reshape(-1)
        for transformer_id in ev_features[graph_ev_positions, 5].round().long().tolist():
            key = (graph_index, int(transformer_id))
            counts[key] = counts.get(key, 0) + 1
        graph_node_start = graph_node_end
    return counts


def _config_timescale_minutes(config_path: str | Path) -> float:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "timescale" not in config:
        raise ValueError("projection diagnostics require config timescale")
    return float(config["timescale"])


def select_action_with_projection_diagnostics(
    policy,
    state,
    *,
    expl_noise: float = 0.0,
    noise_clip: float | None = None,
    metadata: dict[str, object] | None = None,
    episode_index: int = 0,
    episode_seed: int = 0,
    environment_step: int = 0,
) -> ProjectionActionDiagnostics:
    if not all(hasattr(policy, name) for name in ("transformer_voltage", "transformer_phases")):
        raise ValueError("projection diagnostics require hierarchical_transformer_constraint policy")

    state = state.to(policy.device)
    with torch.no_grad():
        raw_action = policy.actor(state)
        candidate_action = policy._add_ev_noise(state, raw_action, expl_noise, noise_clip)
        safe_action, details = project_transformer_feasible_actions(
            state=state,
            full_node_action=candidate_action,
            voltage=policy.transformer_voltage,
            phases=policy.transformer_phases,
            max_action=policy.max_action,
            return_details=True,
        )

    active_ev_indexes = torch.as_tensor(state.ev_indexes, dtype=torch.long, device=safe_action.device)
    if active_ev_indexes.numel() == 0:
        corrected = 0
        active = 0
    else:
        changed = (
            candidate_action[active_ev_indexes, 0] - safe_action[active_ev_indexes, 0]
        ).abs() > PROJECTION_DIAGNOSTIC_TOLERANCE
        corrected = int(torch.count_nonzero(changed).item())
        active = int(active_ev_indexes.numel())

    rows = transformer_step_rows_from_projection_details(
        metadata=metadata or {},
        episode_index=episode_index,
        episode_seed=episode_seed,
        environment_step=environment_step,
        details=details,
        active_ev_count_by_transformer=_active_ev_count_by_transformer(state),
        corrected_ev_decisions=corrected,
    )
    mapped_action = policy._map_to_ev2gym_action(state, safe_action)
    return ProjectionActionDiagnostics(
        raw_action=candidate_action.detach().cpu(),
        safe_action=safe_action.detach().cpu(),
        mapped_action=mapped_action,
        transformer_rows=rows,
        corrected_ev_decisions=corrected,
        active_ev_decisions=active,
    )


def _transformer_invariant(env, inspected_step_index: int) -> tuple[int, float]:
    violations = 0
    max_excess = 0.0
    for transformer in env.transformers:
        actual_power = float(transformer.current_power)
        max_power = float(transformer.max_power[inspected_step_index])
        excess = actual_power - max_power
        if excess > 1e-9:
            violations += 1
            max_excess = max(max_excess, excess)
    return violations, max_excess


def evaluate_projection_diagnostic_episode(
    *,
    policy,
    env,
    seed: int,
    metadata: dict[str, object],
    timestep_minutes: float,
    max_episode_steps: int | None = None,
    deterministic: bool = True,
    eval_expl_noise: float = 0.0,
    episode_index: int = 0,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    state, _reset_info = reset_env_state(env, seed=seed)
    done = False
    environment_step = 0
    transformer_rows: list[dict[str, object]] = []
    corrected_decisions = 0
    active_decisions = 0
    violation_count = 0
    max_excess = 0.0

    while not done:
        action_diagnostics = select_action_with_projection_diagnostics(
            policy,
            state,
            expl_noise=0.0 if deterministic else eval_expl_noise,
            metadata=metadata,
            episode_index=episode_index,
            episode_seed=seed,
            environment_step=environment_step,
        )
        transformer_rows.extend(action_diagnostics.transformer_rows)
        corrected_decisions += action_diagnostics.corrected_ev_decisions
        active_decisions += action_diagnostics.active_ev_decisions
        state, _reward, done, _stats = normalise_step_result(env.step(action_diagnostics.mapped_action))
        inspected_step_index = env.current_step - 1
        step_violations, step_max_excess = _transformer_invariant(env, inspected_step_index)
        violation_count += step_violations
        max_excess = max(max_excess, step_max_excess)
        environment_step += 1

        if max_episode_steps is not None and environment_step >= max_episode_steps:
            done = True

    summary = summarise_projection_episode(
        metadata=metadata,
        episode_index=episode_index,
        episode_seed=seed,
        environment_steps=environment_step,
        transformer_step_rows=transformer_rows,
        timestep_minutes=timestep_minutes,
        corrected_ev_decisions=corrected_decisions,
        active_ev_decisions=active_decisions,
    )
    summary["transformer_feasibility_violation_count"] = violation_count
    summary["transformer_feasibility_max_excess_kw"] = max_excess
    return transformer_rows, summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Diagnostic-only evaluator for hard transformer feasibility projection."
    )
    parser.add_argument("--algorithm", required=True, choices=ALGORITHM_CHOICES)
    parser.add_argument("--scale", required=True, choices=SCALE_CHOICES)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_episodes", type=int, default=1)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", default="transformer_constraint_projection_diagnostics")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max_episode_steps", type=int, default=None)
    parser.add_argument("--deterministic", type=str2bool, default=True)
    parser.add_argument("--eval_expl_noise", type=float, default=0.0)
    parser.add_argument("--eval_seed_offset", type=int, default=100000)
    parser.add_argument("--matrix_job_id", default="")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    canonical_algorithm = normalise_algorithm_label(args.algorithm)
    if canonical_algorithm != "hierarchical_transformer_constraint":
        raise ValueError("projection diagnostics only support hierarchical_transformer_constraint")
    device = resolve_device(args.device)
    checkpoint_prefix = normalise_checkpoint_prefix(args.checkpoint)
    checkpoint_kwargs = load_checkpoint_kwargs(checkpoint_prefix)

    probe_env = make_env(args.config, seed=args.seed)
    action_dim = probe_env.action_space.shape[0]
    max_action = float(probe_env.action_space.high[0])

    validate_checkpoint_identity(checkpoint_prefix, canonical_algorithm)
    policy = create_policy(
        algorithm=canonical_algorithm,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        checkpoint_kwargs=checkpoint_kwargs,
    )
    load_policy_checkpoint(policy, checkpoint_prefix, canonical_algorithm)

    metadata = {
        "matrix_job_id": args.matrix_job_id,
        "scale": args.scale,
        "algorithm": canonical_algorithm,
        "training_seed": args.seed,
        "config": args.config,
        "checkpoint_prefix": str(checkpoint_prefix),
        "run_name": args.run_name,
    }
    timestep_minutes = _config_timescale_minutes(args.config)
    all_transformer_rows = []
    episode_summary_rows = []
    for episode_index in range(args.eval_episodes):
        episode_seed = args.seed + args.eval_seed_offset + episode_index
        env = make_env(args.config, seed=episode_seed)
        transformer_rows, episode_summary = evaluate_projection_diagnostic_episode(
            policy=policy,
            env=env,
            seed=episode_seed,
            metadata=metadata,
            timestep_minutes=timestep_minutes,
            max_episode_steps=args.max_episode_steps,
            deterministic=args.deterministic,
            eval_expl_noise=args.eval_expl_noise,
            episode_index=episode_index,
        )
        all_transformer_rows.extend(transformer_rows)
        episode_summary_rows.append(episode_summary)

    seed_summary = summarise_projection_seed(
        {
            "algorithm": canonical_algorithm,
            "scale": args.scale,
            "training_seed": args.seed,
        },
        episode_summary_rows,
    )
    output_dir = Path(args.output_dir)
    write_csv_rows(
        output_dir / "projection_transformer_step_rows.csv",
        PROJECTION_TRANSFORMER_STEP_COLUMNS,
        all_transformer_rows,
    )
    write_csv_rows(
        output_dir / "projection_episode_summary.csv",
        PROJECTION_EPISODE_SUMMARY_COLUMNS,
        episode_summary_rows,
    )
    write_csv_rows(
        output_dir / "projection_seed_summary.csv",
        PROJECTION_SEED_SUMMARY_COLUMNS,
        [seed_summary],
    )
    print("PROJECTION_DIAGNOSTICS_WRITTEN")
    print(f"PROJECTION_TRANSFORMER_STEP_ROWS={len(all_transformer_rows)}")
    print(f"PROJECTION_EPISODE_ROWS={len(episode_summary_rows)}")
    print(f"OUTPUT_DIR={output_dir}")


if __name__ == "__main__":
    main()
