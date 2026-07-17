#!/usr/bin/env python3
"""Diagnostic actor-runtime breakdown for TD3 EV-GNN policies.

This profiler is instrumentation only. It builds fixed real PublicPST graph
states, reuses those states for all repetitions, and times actor-facing phases
without changing actor, critic, replay-buffer, state, reward, training, or
evaluation semantics.

The forward-plus-backward phases use a deterministic scalar diagnostic loss
derived directly from actor outputs. They are not TD3 training objectives and
they do not perform optimiser updates.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ALGORITHMS = ("actiongnn", "hierarchical")

CSV_FIELDNAMES = [
    "run_name",
    "config",
    "scale",
    "algorithm",
    "batch_size",
    "schedule_index",
    "algorithm_order_position",
    "repetition",
    "status",
    "phase",
    "elapsed_seconds",
    "total_nodes",
    "active_ev_count",
    "charger_count",
    "transformer_count",
    "num_graphs",
    "cpu_thread_count",
    "device",
]


@dataclass
class TimingRecord:
    run_name: str
    config: str
    scale: str
    algorithm: str
    batch_size: int
    repetition: int
    status: str
    phase: str
    elapsed_seconds: float
    total_nodes: int
    active_ev_count: int
    charger_count: int
    transformer_count: int
    num_graphs: int
    cpu_thread_count: int
    device: str
    schedule_index: int = 0
    algorithm_order_position: int = 0

    def to_csv_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["elapsed_seconds"] = f"{self.elapsed_seconds:.9f}"
        return row


def positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed_value


def nonnegative_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed_value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic-only ActionGNN vs hierarchical actor runtime breakdown "
            "on fixed real PublicPST graph states."
        )
    )
    parser.add_argument("--config", required=True, help="EV2Gym PublicPST-compatible config path.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output_dir", "--output-dir", required=True)
    parser.add_argument("--run_name", "--run-name", default=None)
    parser.add_argument(
        "--batch_sizes",
        "--batch-sizes",
        type=positive_int,
        nargs="+",
        default=[1, 64],
        help="Batch sizes to measure; defaults to online batch 1 and replay batch 64.",
    )
    parser.add_argument(
        "--warmup_repetitions",
        "--warmup-repetitions",
        type=nonnegative_int,
        default=3,
    )
    parser.add_argument(
        "--measured_repetitions",
        "--measured-repetitions",
        type=positive_int,
        default=10,
    )
    parser.add_argument("--replay_buffer_size", "--replay-buffer-size", type=positive_int, default=100000)
    parser.add_argument("--max_collection_steps", "--max-collection-steps", type=positive_int, default=5000)
    parser.add_argument("--log_every", "--log-every", type=nonnegative_int, default=0)
    parser.add_argument("--fx_dim", "--fx-dim", type=positive_int, default=32)
    parser.add_argument("--fx_GNN_hidden_dim", "--fx-GNN-hidden-dim", type=positive_int, default=64)
    parser.add_argument("--mlp_hidden_dim", "--mlp-hidden-dim", type=positive_int, default=512)
    parser.add_argument("--actor_num_gcn_layers", "--actor-num-gcn-layers", type=positive_int, default=3)
    parser.add_argument("--critic_num_gcn_layers", "--critic-num-gcn-layers", type=positive_int, default=3)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not Path(args.config).exists():
        parser.error(f"--config does not exist: {args.config}")
    args.batch_sizes = list(dict.fromkeys(args.batch_sizes))
    if args.replay_buffer_size < max(args.batch_sizes):
        parser.error("--replay_buffer_size must be at least the largest requested batch size")
    return args


def time_callable(callable_object: Callable[[], Any]) -> tuple[Any, float]:
    start_time = time.perf_counter()
    result = callable_object()
    elapsed_seconds = time.perf_counter() - start_time
    return result, elapsed_seconds


def graph_counts(state) -> dict[str, int]:
    sample_node_length = getattr(state, "sample_node_length", None)
    if sample_node_length is None:
        total_nodes = int(len(getattr(state, "node_types", [])))
        num_graphs = 1
    else:
        sample_lengths = [int(length) for length in sample_node_length]
        total_nodes = int(sum(sample_lengths))
        num_graphs = int(len(sample_lengths))

    return {
        "total_nodes": total_nodes,
        "active_ev_count": int(len(getattr(state, "ev_indexes", []))),
        "charger_count": int(len(getattr(state, "cs_indexes", []))),
        "transformer_count": int(len(getattr(state, "tr_indexes", []))),
        "num_graphs": num_graphs,
    }


REPRESENTATIVE_SINGLE_STATE_SELECTION = "max_active_ev_then_total_nodes_then_earliest"
GRAPH_COUNT_KEYS = (
    "total_nodes",
    "active_ev_count",
    "charger_count",
    "transformer_count",
    "num_graphs",
)


def select_representative_single_state(states: list[Any]) -> Any:
    if not states:
        raise ValueError("At least one candidate state is required.")

    representative_state = states[0]
    representative_rank = (
        graph_counts(representative_state)["active_ev_count"],
        graph_counts(representative_state)["total_nodes"],
    )
    for candidate_state in states[1:]:
        candidate_counts = graph_counts(candidate_state)
        candidate_rank = (
            candidate_counts["active_ev_count"],
            candidate_counts["total_nodes"],
        )
        if candidate_rank > representative_rank:
            representative_state = candidate_state
            representative_rank = candidate_rank
    return representative_state


def should_finish_collection(replay_threshold_reached: bool, episode_done: bool) -> bool:
    return bool(replay_threshold_reached and episode_done)


def update_representative_single_state(
    representative_state,
    candidate_state,
    collection_step: int,
    episode_index: int,
    representative_metadata: dict[str, int] | None = None,
) -> tuple[Any, dict[str, int]]:
    candidate_metadata = {
        "representative_state_collection_step": int(collection_step),
        "representative_state_episode_index": int(episode_index),
    }
    if representative_state is None:
        return candidate_state, candidate_metadata

    selected_state = select_representative_single_state([representative_state, candidate_state])
    if selected_state is candidate_state:
        return candidate_state, candidate_metadata
    return representative_state, dict(representative_metadata or {})


def get_cpu_thread_count() -> int:
    import torch

    return int(torch.get_num_threads())


def make_record(
    args: argparse.Namespace,
    algorithm: str,
    batch_size: int,
    repetition: int,
    status: str,
    phase: str,
    elapsed_seconds: float,
    state,
    device: str,
    schedule_index: int = 0,
    algorithm_order_position: int = 0,
) -> TimingRecord:
    counts = graph_counts(state)
    return TimingRecord(
        run_name=args.run_name,
        config=args.config,
        scale=Path(args.config).stem,
        algorithm=algorithm,
        batch_size=int(batch_size),
        repetition=int(repetition),
        status=status,
        phase=phase,
        elapsed_seconds=float(elapsed_seconds),
        total_nodes=counts["total_nodes"],
        active_ev_count=counts["active_ev_count"],
        charger_count=counts["charger_count"],
        transformer_count=counts["transformer_count"],
        num_graphs=counts["num_graphs"],
        cpu_thread_count=get_cpu_thread_count(),
        device=device,
        schedule_index=int(schedule_index),
        algorithm_order_position=int(algorithm_order_position),
    )


def calculate_summary_statistics(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "total": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "stdev": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p95": 0.0,
        }

    sorted_values = sorted(float(value) for value in values)
    p95_index = max(0, math.ceil(0.95 * len(sorted_values)) - 1)
    return {
        "count": int(len(sorted_values)),
        "total": float(sum(sorted_values)),
        "mean": float(statistics.mean(sorted_values)),
        "median": float(statistics.median(sorted_values)),
        "stdev": float(statistics.stdev(sorted_values)) if len(sorted_values) > 1 else 0.0,
        "min": float(sorted_values[0]),
        "max": float(sorted_values[-1]),
        "p95": float(sorted_values[p95_index]),
    }


def build_json_summary(records: list[TimingRecord], metadata: dict[str, Any]) -> dict[str, Any]:
    grouped_values: dict[str, dict[str, dict[str, list[float]]]] = {}
    for record in records:
        if record.status != "measured":
            continue
        batch_key = f"batch_{record.batch_size}"
        grouped_values.setdefault(record.algorithm, {}).setdefault(batch_key, {}).setdefault(
            record.phase,
            [],
        ).append(float(record.elapsed_seconds))

    phase_summary = {
        algorithm: {
            batch_key: {
                phase: calculate_summary_statistics(values)
                for phase, values in sorted(phase_values.items())
            }
            for batch_key, phase_values in sorted(batch_values.items())
        }
        for algorithm, batch_values in sorted(grouped_values.items())
    }

    return {
        "metadata": dict(metadata),
        "phases": phase_summary,
    }


def write_outputs(
    records: list[TimingRecord],
    summary: dict[str, Any],
    output_dir: str | Path,
    run_name: str,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / f"{run_name}_hierarchical_actor_breakdown_steps.csv"
    json_path = output_path / f"{run_name}_hierarchical_actor_breakdown_summary.json"

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_csv_row())

    with json_path.open("w") as json_file:
        json.dump(summary, json_file, indent=2, sort_keys=True)
        json_file.write("\n")

    return csv_path, json_path


def snapshot_parameters(module) -> dict[str, Any]:
    return {
        parameter_name: parameter.detach().clone()
        for parameter_name, parameter in module.named_parameters()
    }


def parameters_match_snapshot(module, snapshot: dict[str, Any]) -> bool:
    import torch

    current_parameters = dict(module.named_parameters())
    if set(current_parameters) != set(snapshot):
        return False
    for parameter_name, expected_value in snapshot.items():
        if not torch.equal(current_parameters[parameter_name].detach().cpu(), expected_value.cpu()):
            return False
    return True


def assert_finite_gradients(module) -> None:
    import torch

    missing_gradients = []
    nonfinite_gradients = []
    for parameter_name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing_gradients.append(parameter_name)
            continue
        if not torch.isfinite(parameter.grad).all():
            nonfinite_gradients.append(parameter_name)

    if missing_gradients or nonfinite_gradients:
        raise AssertionError(
            "Invalid diagnostic actor gradients. "
            f"missing={missing_gradients}, nonfinite={nonfinite_gradients}"
        )


def assert_non_ev_action_rows_zero(full_node_action, state) -> None:
    import torch

    total_nodes = graph_counts(state)["total_nodes"]
    reshaped_action = full_node_action.reshape(-1, 1)
    active_ev_node_indexes = torch.as_tensor(
        getattr(state, "ev_indexes", []),
        dtype=torch.long,
        device=reshaped_action.device,
    ).reshape(-1)
    non_ev_mask = torch.ones(total_nodes, dtype=torch.bool, device=reshaped_action.device)
    if active_ev_node_indexes.numel() > 0:
        non_ev_mask[active_ev_node_indexes] = False

    non_ev_actions = reshaped_action[non_ev_mask]
    if not torch.allclose(non_ev_actions, torch.zeros_like(non_ev_actions)):
        raise AssertionError("Non-EV action rows must remain zero.")


def assert_hierarchical_forward_matches_breakdown(complete_output, decomposed_output) -> None:
    import torch

    if complete_output.shape != decomposed_output.shape:
        raise AssertionError(
            "Decomposed hierarchical actor forward shape mismatch: "
            f"complete={tuple(complete_output.shape)}, decomposed={tuple(decomposed_output.shape)}"
        )
    if complete_output.dtype != decomposed_output.dtype:
        raise AssertionError(
            "Decomposed hierarchical actor forward dtype mismatch: "
            f"complete={complete_output.dtype}, decomposed={decomposed_output.dtype}"
        )
    if complete_output.device != decomposed_output.device:
        raise AssertionError(
            "Decomposed hierarchical actor forward device mismatch: "
            f"complete={complete_output.device}, decomposed={decomposed_output.device}"
        )

    try:
        torch.testing.assert_close(
            complete_output,
            decomposed_output,
            atol=1e-6,
            rtol=1e-5,
        )
    except AssertionError as close_error:
        raise AssertionError(
            "Decomposed hierarchical actor forward differs from complete actor forward."
        ) from close_error


def assert_matched_algorithm_graph_counts(records: list[TimingRecord]) -> None:
    counts_by_schedule: dict[tuple[int, int], dict[str, tuple[int, ...]]] = {}
    actor_forward_phases = {
        "actiongnn_actor_forward",
        "hierarchical_actor_forward",
    }
    for record in records:
        if record.phase not in actor_forward_phases:
            continue
        schedule_key = (record.batch_size, record.schedule_index)
        record_counts = tuple(int(getattr(record, key)) for key in GRAPH_COUNT_KEYS)
        counts_by_schedule.setdefault(schedule_key, {})[record.algorithm] = record_counts

    for schedule_key, algorithm_counts in counts_by_schedule.items():
        if set(algorithm_counts) != set(ALGORITHMS):
            continue
        if algorithm_counts["actiongnn"] != algorithm_counts["hierarchical"]:
            raise AssertionError(
                "Mismatched graph counts for batch_size="
                f"{schedule_key[0]}, schedule_index={schedule_key[1]}: "
                f"{algorithm_counts}"
            )


def run_actor_forward_backward(actor, state, algorithm: str):
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unsupported algorithm for diagnostic backward: {algorithm}")

    actor.zero_grad(set_to_none=True)
    actor_output = actor(state)
    diagnostic_loss = actor_output.reshape(-1).float().sum()
    if not diagnostic_loss.requires_grad:
        raise RuntimeError("Diagnostic actor loss is not differentiable for this state.")
    diagnostic_loss.backward()
    return actor_output, diagnostic_loss.detach()


def measure_hierarchical_forward_breakdown(actor, state) -> tuple[Any, dict[str, float]]:
    import torch

    phase_timings: dict[str, float] = {}

    (embedded_node_features, edge_index), elapsed_seconds = time_callable(
        lambda: actor._prepare_features(state)
    )
    phase_timings["hierarchical_prepare_features"] = elapsed_seconds

    def prepare_compose_inputs():
        return {
            "ev_features": actor._tensor(state.ev_features, torch.float32),
            "cs_features": actor._tensor(state.cs_features, torch.float32),
            "tr_features": actor._tensor(state.tr_features, torch.float32),
            "active_ev_node_indexes": actor._index_tensor(state.ev_indexes),
            "charger_node_indexes": actor._index_tensor(state.cs_indexes),
            "transformer_node_indexes": actor._index_tensor(state.tr_indexes),
        }

    compose_inputs, elapsed_seconds = time_callable(prepare_compose_inputs)
    phase_timings["hierarchical_prepare_compose_inputs"] = elapsed_seconds

    node_embeddings, elapsed_seconds = time_callable(
        lambda: actor._encode_nodes(embedded_node_features, edge_index)
    )
    phase_timings["hierarchical_encode_nodes"] = elapsed_seconds

    (full_node_action, _), elapsed_seconds = time_callable(
        lambda: actor._compose_full_node_action(
            state=state,
            node_embeddings=node_embeddings,
            **compose_inputs,
        )
    )
    phase_timings["hierarchical_compose_full_node_action"] = elapsed_seconds
    return full_node_action, phase_timings


def reset_env_state(env, seed: int):
    from utils.ev2gym_training_utils import reset_env

    reset_result = reset_env(env, seed=seed)
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        return reset_result
    return reset_result, {}


def make_policies(args: argparse.Namespace, action_dim: int, max_action: float, device: str):
    from TD3.TD3_ActionGNN_Controlled import TD3_ActionGNN
    from TD3.TD3_HierarchicalActionGNN import TD3_HierarchicalActionGNN
    from utils.state_public_pst_gnn import PublicPST_GNN

    common_kwargs = {
        "action_dim": action_dim,
        "max_action": max_action,
        "fx_node_sizes": PublicPST_GNN.node_sizes,
        "discount": 0.99,
        "tau": 0.005,
        "policy_noise": 0.2 * max_action,
        "noise_clip": 0.5 * max_action,
        "policy_freq": 2,
        "fx_dim": args.fx_dim,
        "fx_GNN_hidden_dim": args.fx_GNN_hidden_dim,
        "mlp_hidden_dim": args.mlp_hidden_dim,
        "lr": 3e-4,
        "discrete_actions": 1,
        "actor_num_gcn_layers": args.actor_num_gcn_layers,
        "critic_num_gcn_layers": args.critic_num_gcn_layers,
        "device": device,
    }
    return {
        "actiongnn": TD3_ActionGNN(**common_kwargs),
        "hierarchical": TD3_HierarchicalActionGNN(**common_kwargs),
    }


def zero_full_node_action_for_state(state):
    import torch

    return torch.zeros((graph_counts(state)["total_nodes"], 1), dtype=torch.float32)


def collect_until_threshold_episode_complete(
    initial_state,
    target_replay_size: int,
    max_collection_steps: int,
    step_state: Callable[[Any], tuple[Any, float, bool, Any]],
    reset_episode: Callable[[int], Any],
    add_active_transition: Callable[[Any, Any, float, bool], int | None],
    log_every: int = 0,
) -> dict[str, Any]:
    state = initial_state
    representative_single_state = None
    representative_metadata: dict[str, int] = {}
    replay_threshold_reached_at_step = None
    replay_threshold_episode_index = None
    completed_collection_episodes = 0
    collected_active_transition_count = 0
    episode_index = 0

    for collection_step in range(1, int(max_collection_steps) + 1):
        state_has_active_ev = len(getattr(state, "ev_indexes", [])) > 0
        if state_has_active_ev:
            representative_single_state, representative_metadata = update_representative_single_state(
                representative_state=representative_single_state,
                candidate_state=state,
                collection_step=collection_step,
                episode_index=episode_index,
                representative_metadata=representative_metadata,
            )

        next_state, reward, done, _ = step_state(state)

        if state_has_active_ev:
            retained_transition_count = add_active_transition(state, next_state, reward, done)
            collected_active_transition_count += 1
            if retained_transition_count is None:
                retained_transition_count = collected_active_transition_count
            if (
                replay_threshold_reached_at_step is None
                and int(retained_transition_count) >= int(target_replay_size)
            ):
                replay_threshold_reached_at_step = collection_step
                replay_threshold_episode_index = episode_index

        if done:
            completed_collection_episodes += 1

        replay_threshold_reached = replay_threshold_reached_at_step is not None
        if should_finish_collection(replay_threshold_reached, done):
            return {
                "representative_single_state": representative_single_state,
                "representative_single_state_counts": graph_counts(representative_single_state),
                "representative_single_state_selection": REPRESENTATIVE_SINGLE_STATE_SELECTION,
                "collection_steps_completed": collection_step,
                "collected_active_transition_count": collected_active_transition_count,
                "completed_collection_episodes": completed_collection_episodes,
                "replay_threshold_reached_at_step": replay_threshold_reached_at_step,
                "replay_threshold_episode_index": replay_threshold_episode_index,
                "collection_termination_reason": "completed_threshold_episode",
                **representative_metadata,
            }

        if done:
            episode_index += 1
            state = reset_episode(episode_index)
        else:
            state = next_state

        if log_every > 0 and collection_step % log_every == 0:
            print(
                f"[collect] step={collection_step} "
                f"active_transitions={collected_active_transition_count}/{target_replay_size}"
            )

    if replay_threshold_reached_at_step is not None:
        raise RuntimeError(
            "Replay threshold reached at step "
            f"{replay_threshold_reached_at_step} in episode "
            f"{replay_threshold_episode_index}, but the threshold episode did not "
            f"terminate before --max_collection_steps={max_collection_steps}."
        )
    if representative_single_state is None:
        raise RuntimeError("Could not collect any active-EV PublicPST graph state.")
    raise RuntimeError(
        "Could not collect enough active-EV transitions. "
        f"Collected {collected_active_transition_count}, required {target_replay_size}."
    )


def collect_public_pst_replay_data(args: argparse.Namespace, device: str):
    import numpy as np

    from utils.ev2gym_training_utils import (
        make_env,
        normalise_step_result,
        set_global_seed,
    )
    from utils.replay_buffer_actiongnn import ActionGNN_ReplayBuffer

    set_global_seed(args.seed)
    env = make_env(args.config, seed=args.seed)
    action_dim = int(env.action_space.shape[0])
    max_action = float(env.action_space.high[0])
    replay_buffer = ActionGNN_ReplayBuffer(
        action_dim=action_dim,
        max_size=args.replay_buffer_size,
        device=device,
    )

    zero_mapped_action = np.zeros(action_dim, dtype=np.float32)
    target_replay_size = max(args.batch_sizes)
    initial_state, _ = reset_env_state(env, seed=args.seed)

    def step_state(state):
        return normalise_step_result(env.step(zero_mapped_action))

    def reset_collection_episode(episode_index):
        reset_state, _ = reset_env_state(env, seed=args.seed + episode_index)
        return reset_state

    def add_active_transition(state, next_state, reward, done):
        state_for_replay = copy.deepcopy(state)
        replay_buffer.add(
            state_for_replay,
            zero_full_node_action_for_state(state_for_replay),
            copy.deepcopy(next_state),
            reward,
            done,
        )
        return replay_buffer.size

    collection_data = collect_until_threshold_episode_complete(
        initial_state=initial_state,
        target_replay_size=target_replay_size,
        max_collection_steps=args.max_collection_steps,
        step_state=step_state,
        reset_episode=reset_collection_episode,
        add_active_transition=add_active_transition,
        log_every=args.log_every,
    )

    return {
        "env": env,
        "action_dim": action_dim,
        "max_action": max_action,
        "replay_buffer": replay_buffer,
        **collection_data,
    }


def fixed_replay_batches(args: argparse.Namespace, replay_buffer) -> dict[int, Any]:
    import numpy as np

    batches = {}
    for batch_size in args.batch_sizes:
        np.random.seed(args.seed + 10_000 + batch_size)
        batches[batch_size] = replay_buffer.sample(batch_size)
    return batches


def algorithm_order_for_schedule_index(schedule_index: int) -> tuple[str, str]:
    if int(schedule_index) % 2 == 0:
        return ("actiongnn", "hierarchical")
    return ("hierarchical", "actiongnn")


def repetition_schedule(args: argparse.Namespace) -> list[tuple[int, str, int]]:
    schedule = [
        (schedule_index, "warmup", schedule_index)
        for schedule_index in range(args.warmup_repetitions)
    ]
    measured_start_index = len(schedule)
    schedule.extend(
        (measured_start_index + repetition, "measured", repetition)
        for repetition in range(args.measured_repetitions)
    )
    return schedule


def run_measurements(args: argparse.Namespace) -> tuple[list[TimingRecord], dict[str, Any]]:
    import numpy as np
    import torch
    import torch_geometric

    from utils.ev2gym_training_utils import resolve_device, set_global_seed

    if args.run_name is None:
        args.run_name = f"{Path(args.config).stem}_hierarchical_actor_breakdown_seed{args.seed}"

    device = resolve_device(args.device)
    set_global_seed(args.seed)

    collected_data = collect_public_pst_replay_data(args, device=device)
    replay_buffer = collected_data["replay_buffer"]
    fixed_batches = fixed_replay_batches(args, replay_buffer)
    policies = make_policies(
        args=args,
        action_dim=collected_data["action_dim"],
        max_action=collected_data["max_action"],
        device=device,
    )
    actor_parameter_snapshots = {
        algorithm: snapshot_parameters(policy.actor)
        for algorithm, policy in policies.items()
    }

    records: list[TimingRecord] = []
    total_wall_start = time.perf_counter()

    for batch_size in args.batch_sizes:
        actor_state = (
            collected_data["representative_single_state"]
            if batch_size == 1
            else fixed_batches[batch_size][0]
        )

        for schedule_index, status, repetition in repetition_schedule(args):
            actor_state_counts_by_algorithm = {}
            for algorithm_order_position, algorithm in enumerate(
                algorithm_order_for_schedule_index(schedule_index)
            ):
                policy = policies[algorithm]
                actor_state_counts_by_algorithm[algorithm] = tuple(
                    graph_counts(actor_state)[key]
                    for key in GRAPH_COUNT_KEYS
                )

                sample_seed = args.seed + batch_size * 100_000 + schedule_index
                np.random.seed(sample_seed)
                replay_sample, elapsed_seconds = time_callable(
                    lambda: replay_buffer.sample(batch_size)
                )
                records.append(
                    make_record(
                        args=args,
                        algorithm=algorithm,
                        batch_size=batch_size,
                        repetition=repetition,
                        status=status,
                        phase="replay_buffer_sample",
                        elapsed_seconds=elapsed_seconds,
                        state=replay_sample[0],
                        device=device,
                        schedule_index=schedule_index,
                        algorithm_order_position=algorithm_order_position,
                    )
                )

                if algorithm == "actiongnn":
                    with torch.no_grad():
                        _, elapsed_seconds = time_callable(lambda: policy.actor(actor_state))
                    records.append(
                        make_record(
                            args,
                            algorithm,
                            batch_size,
                            repetition,
                            status,
                            "actiongnn_actor_forward",
                            elapsed_seconds,
                            actor_state,
                            device,
                            schedule_index=schedule_index,
                            algorithm_order_position=algorithm_order_position,
                        )
                    )

                    _, elapsed_seconds = time_callable(
                        lambda: run_actor_forward_backward(policy.actor, actor_state, algorithm)
                    )
                    assert_finite_gradients(policy.actor)
                    records.append(
                        make_record(
                            args,
                            algorithm,
                            batch_size,
                            repetition,
                            status,
                            "actiongnn_actor_forward_backward",
                            elapsed_seconds,
                            actor_state,
                            device,
                            schedule_index=schedule_index,
                            algorithm_order_position=algorithm_order_position,
                        )
                    )
                    continue

                with torch.no_grad():
                    full_node_action, elapsed_seconds = time_callable(
                        lambda: policy.actor(actor_state)
                    )
                records.append(
                    make_record(
                        args,
                        algorithm,
                        batch_size,
                        repetition,
                        status,
                        "hierarchical_actor_forward",
                        elapsed_seconds,
                        actor_state,
                        device,
                        schedule_index=schedule_index,
                        algorithm_order_position=algorithm_order_position,
                    )
                )
                assert_non_ev_action_rows_zero(full_node_action, actor_state)

                with torch.no_grad():
                    breakdown_action, phase_timings = measure_hierarchical_forward_breakdown(
                        policy.actor,
                        actor_state,
                )
                assert_non_ev_action_rows_zero(breakdown_action, actor_state)
                assert_hierarchical_forward_matches_breakdown(full_node_action, breakdown_action)
                for phase, phase_elapsed_seconds in phase_timings.items():
                    records.append(
                        make_record(
                            args,
                            algorithm,
                            batch_size,
                            repetition,
                            status,
                            phase,
                            phase_elapsed_seconds,
                            actor_state,
                            device,
                            schedule_index=schedule_index,
                            algorithm_order_position=algorithm_order_position,
                        )
                    )

                if batch_size == 1:
                    _, elapsed_seconds = time_callable(
                        lambda: policy._map_to_ev2gym_action(actor_state, full_node_action)
                    )
                    records.append(
                        make_record(
                            args,
                            algorithm,
                            batch_size,
                            repetition,
                            status,
                            "hierarchical_map_to_ev2gym_action",
                            elapsed_seconds,
                            actor_state,
                            device,
                            schedule_index=schedule_index,
                            algorithm_order_position=algorithm_order_position,
                        )
                    )

                _, elapsed_seconds = time_callable(
                    lambda: run_actor_forward_backward(policy.actor, actor_state, algorithm)
                )
                assert_finite_gradients(policy.actor)
                records.append(
                    make_record(
                        args,
                        algorithm,
                        batch_size,
                        repetition,
                        status,
                        "hierarchical_actor_forward_backward",
                        elapsed_seconds,
                        actor_state,
                        device,
                        schedule_index=schedule_index,
                        algorithm_order_position=algorithm_order_position,
                    )
                )

            if actor_state_counts_by_algorithm["actiongnn"] != actor_state_counts_by_algorithm["hierarchical"]:
                raise AssertionError(
                    "Mismatched graph counts for actor timing in batch_size="
                    f"{batch_size}, schedule_index={schedule_index}: "
                    f"{actor_state_counts_by_algorithm}"
                )

    assert_matched_algorithm_graph_counts(records)

    for algorithm, policy in policies.items():
        if not parameters_match_snapshot(policy.actor, actor_parameter_snapshots[algorithm]):
            raise RuntimeError(f"{algorithm} actor parameters changed during diagnostic profiling.")

    total_wall_time_seconds = time.perf_counter() - total_wall_start
    metadata = {
        "run_name": args.run_name,
        "config": args.config,
        "scale": Path(args.config).stem,
        "seed": int(args.seed),
        "batch_sizes": [int(batch_size) for batch_size in args.batch_sizes],
        "warmup_repetitions": int(args.warmup_repetitions),
        "measured_repetitions": int(args.measured_repetitions),
        "device": device,
        "action_dim": int(collected_data["action_dim"]),
        "max_action": float(collected_data["max_action"]),
        "replay_buffer_size": int(replay_buffer.size),
        "collection_steps_completed": int(collected_data["collection_steps_completed"]),
        "collected_active_transition_count": int(collected_data["collected_active_transition_count"]),
        "completed_collection_episodes": int(collected_data["completed_collection_episodes"]),
        "replay_threshold_reached_at_step": int(collected_data["replay_threshold_reached_at_step"]),
        "replay_threshold_episode_index": int(collected_data["replay_threshold_episode_index"]),
        "representative_state_collection_step": int(collected_data["representative_state_collection_step"]),
        "representative_state_episode_index": int(collected_data["representative_state_episode_index"]),
        "representative_single_state_counts": graph_counts(
            collected_data["representative_single_state"]
        ),
        "representative_single_state_selection": REPRESENTATIVE_SINGLE_STATE_SELECTION,
        "collection_termination_reason": collected_data["collection_termination_reason"],
        "total_wall_time_seconds": float(total_wall_time_seconds),
        "diagnostic_backward_note": (
            "Forward-plus-backward phases use sum(actor_output) as a deterministic "
            "diagnostic scalar loss. No critic objective, target network, optimiser "
            "step, checkpoint, or TD3 update is performed."
        ),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_geometric_version": torch_geometric.__version__,
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
        "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS", ""),
    }
    return records, metadata


def main() -> None:
    args = parse_args()
    records, metadata = run_measurements(args)
    summary = build_json_summary(records, metadata)
    csv_path, json_path = write_outputs(
        records=records,
        summary=summary,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )

    print("---------------------------------------")
    print("Hierarchical actor breakdown profile complete")
    print(f"Run name: {args.run_name}")
    print(f"Config: {args.config}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Measured repetitions: {args.measured_repetitions}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print("---------------------------------------")


if __name__ == "__main__":
    main()
