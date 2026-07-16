#!/usr/bin/env python3
"""Runtime profiler for short TD3 EV-GNN training-like runs.

This runner is diagnostic-only: it reuses the existing policy, graph-state,
environment, and replay-buffer components, and only limits how many loop steps
are executed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ALGORITHM_CHOICES = ("actiongnn", "hierarchical")

CSV_FIELDNAMES = [
    "run_name",
    "algorithm",
    "config",
    "seed",
    "step",
    "episode_step",
    "phase",
    "elapsed_seconds",
    "reward",
    "done",
    "replay_size",
    "action_dim",
    "num_nodes",
    "num_evs",
    "num_cs",
    "num_tr",
    "batch_size",
    "device",
    "policy_train_called",
    "critic_loss",
    "actor_loss",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M3-only runtime profiling runner for short TD3 EV-GNN loops."
    )
    parser.add_argument("--algorithm", required=True, choices=sorted(ALGORITHM_CHOICES))
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--profile_steps", type=int, default=2000)
    parser.add_argument("--start_timesteps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--replay_buffer_size", type=int, default=100000)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--eval_episodes", type=int, default=0)
    parser.add_argument("--eval_freq", type=int, default=0)
    parser.add_argument("--max_episode_steps", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=100)
    return parser.parse_args()


def reset_env_state(env, seed):
    from utils.ev2gym_training_utils import reset_env

    reset_result = reset_env(env, seed=seed)
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        return reset_result
    return reset_result, {}


def graph_counts(state) -> dict[str, int]:
    return {
        "num_nodes": int(len(getattr(state, "node_types", []))),
        "num_evs": int(len(getattr(state, "ev_indexes", []))),
        "num_cs": int(len(getattr(state, "cs_indexes", []))),
        "num_tr": int(len(getattr(state, "tr_indexes", []))),
    }


def make_policy(algorithm: str, action_dim: int, max_action: float, device: str):
    from TD3.TD3_ActionGNN_Controlled import TD3_ActionGNN
    from TD3.TD3_HierarchicalActionGNN import TD3_HierarchicalActionGNN
    from utils.state_public_pst_gnn import PublicPST_GNN

    policy_classes = {
        "actiongnn": TD3_ActionGNN,
        "hierarchical": TD3_HierarchicalActionGNN,
    }
    policy_class = policy_classes[algorithm]
    return policy_class(
        action_dim=action_dim,
        max_action=max_action,
        fx_node_sizes=PublicPST_GNN.node_sizes,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2 * max_action,
        noise_clip=0.5 * max_action,
        policy_freq=2,
        fx_dim=32,
        fx_GNN_hidden_dim=64,
        mlp_hidden_dim=512,
        lr=3e-4,
        discrete_actions=1,
        actor_num_gcn_layers=3,
        critic_num_gcn_layers=3,
        device=device,
    )


def write_phase_row(
    writer: csv.DictWriter,
    args: argparse.Namespace,
    step: int,
    episode_step: int,
    phase: str,
    elapsed_seconds: float,
    reward,
    done,
    replay_size: int,
    action_dim: int,
    state,
    device: str,
    policy_train_called: bool = False,
    critic_loss=None,
    actor_loss=None,
) -> None:
    row = {
        "run_name": args.run_name,
        "algorithm": args.algorithm,
        "config": args.config,
        "seed": args.seed,
        "step": step,
        "episode_step": episode_step,
        "phase": phase,
        "elapsed_seconds": float(elapsed_seconds),
        "reward": "" if reward is None else float(reward),
        "done": "" if done is None else bool(done),
        "replay_size": int(replay_size),
        "action_dim": int(action_dim),
        "batch_size": int(args.batch_size),
        "device": device,
        "policy_train_called": bool(policy_train_called),
        "critic_loss": "" if critic_loss is None else float(critic_loss),
        "actor_loss": "" if actor_loss is None else float(actor_loss),
    }
    row.update(graph_counts(state))
    writer.writerow(row)


def timed_phase(phase_totals: dict[str, float], phase_counts: dict[str, int], phase: str, start_time: float) -> float:
    elapsed_seconds = time.perf_counter() - start_time
    phase_totals[phase] = phase_totals.get(phase, 0.0) + elapsed_seconds
    phase_counts[phase] = phase_counts.get(phase, 0) + 1
    return elapsed_seconds


def safe_mean(total: float, count: int) -> float:
    if count == 0:
        return 0.0
    return float(total / count)


def build_summary(
    args: argparse.Namespace,
    action_dim: int,
    device: str,
    total_wall_time_seconds: float,
    phase_totals: dict[str, float],
    phase_counts: dict[str, int],
    replay_buffer,
) -> dict:
    import torch

    number_of_env_steps = phase_counts.get("env_step", 0)
    number_of_policy_train_calls = phase_counts.get("policy_train", 0)
    return {
        "run_name": args.run_name,
        "algorithm": args.algorithm,
        "config": args.config,
        "seed": args.seed,
        "profile_steps": args.profile_steps,
        "start_timesteps": args.start_timesteps,
        "batch_size": args.batch_size,
        "replay_buffer_size": args.replay_buffer_size,
        "action_dim": action_dim,
        "device": device,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "total_wall_time_seconds": float(total_wall_time_seconds),
        "total_env_step_time_seconds": float(phase_totals.get("env_step", 0.0)),
        "total_select_action_time_seconds": float(phase_totals.get("select_action", 0.0)),
        "total_policy_train_time_seconds": float(phase_totals.get("policy_train", 0.0)),
        "total_replay_add_time_seconds": float(phase_totals.get("replay_add", 0.0)),
        "mean_env_step_time_seconds": safe_mean(
            phase_totals.get("env_step", 0.0),
            number_of_env_steps,
        ),
        "mean_select_action_time_seconds": safe_mean(
            phase_totals.get("select_action", 0.0),
            phase_counts.get("select_action", 0),
        ),
        "mean_policy_train_time_seconds": safe_mean(
            phase_totals.get("policy_train", 0.0),
            number_of_policy_train_calls,
        ),
        "mean_total_loop_step_time_seconds": safe_mean(
            phase_totals.get("total_loop_step", 0.0),
            phase_counts.get("total_loop_step", 0),
        ),
        "number_of_policy_train_calls": int(number_of_policy_train_calls),
        "number_of_env_steps": int(number_of_env_steps),
        "final_replay_size": int(replay_buffer.size),
        "total_env_reset_time_seconds": float(phase_totals.get("env_reset", 0.0)),
        "total_replay_sample_or_train_guard_time_seconds": float(
            phase_totals.get("replay_sample_or_train_guard", 0.0)
        ),
        "eval_episodes": int(args.eval_episodes),
        "eval_freq": int(args.eval_freq),
        "max_episode_steps": args.max_episode_steps,
        "replay_sample_timed_separately": False,
        "policy_train_timing_note": (
            "policy.train(...) internally calls replay_buffer.sample(...); "
            "policy_train is therefore combined replay sampling plus TD3 update cost."
        ),
    }


def main() -> None:
    args = parse_args()

    from utils.ev2gym_training_utils import (
        make_env,
        normalise_step_result,
        resolve_device,
        set_global_seed,
    )
    from utils.replay_buffer_actiongnn import ActionGNN_ReplayBuffer

    if args.profile_steps <= 0:
        raise ValueError("--profile_steps must be positive.")
    if args.start_timesteps < 0:
        raise ValueError("--start_timesteps must be non-negative.")

    device = resolve_device(args.device)
    set_global_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.run_name}_runtime_profile_steps.csv"
    summary_path = output_dir / f"{args.run_name}_runtime_profile_summary.json"

    env = make_env(args.config, seed=args.seed)
    action_dim = int(env.action_space.shape[0])
    max_action = float(env.action_space.high[0])
    policy = make_policy(args.algorithm, action_dim=action_dim, max_action=max_action, device=device)
    replay_buffer = ActionGNN_ReplayBuffer(
        action_dim=action_dim,
        max_size=args.replay_buffer_size,
        device=device,
    )

    phase_totals: dict[str, float] = {}
    phase_counts: dict[str, int] = {}
    episode_index = 0
    episode_step = 0
    total_wall_start = time.perf_counter()

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        reset_start = time.perf_counter()
        state, _ = reset_env_state(env, seed=args.seed)
        reset_elapsed = timed_phase(phase_totals, phase_counts, "env_reset", reset_start)
        write_phase_row(
            writer=writer,
            args=args,
            step=0,
            episode_step=0,
            phase="env_reset",
            elapsed_seconds=reset_elapsed,
            reward=None,
            done=False,
            replay_size=replay_buffer.size,
            action_dim=action_dim,
            state=state,
            device=device,
        )
        csv_file.flush()

        for timestep in range(args.profile_steps):
            step = timestep + 1
            loop_start = time.perf_counter()
            episode_step += 1

            select_start = time.perf_counter()
            mapped_action, node_action = policy.select_action(
                state,
                expl_noise=0.1,
                return_mapped_action=True,
            )
            select_elapsed = timed_phase(phase_totals, phase_counts, "select_action", select_start)
            write_phase_row(
                writer=writer,
                args=args,
                step=step,
                episode_step=episode_step,
                phase="select_action",
                elapsed_seconds=select_elapsed,
                reward=None,
                done=None,
                replay_size=replay_buffer.size,
                action_dim=action_dim,
                state=state,
                device=device,
            )

            env_step_start = time.perf_counter()
            next_state, reward, done, _ = normalise_step_result(env.step(mapped_action))
            if args.max_episode_steps is not None and episode_step >= args.max_episode_steps:
                done = True
            env_step_elapsed = timed_phase(phase_totals, phase_counts, "env_step", env_step_start)
            write_phase_row(
                writer=writer,
                args=args,
                step=step,
                episode_step=episode_step,
                phase="env_step",
                elapsed_seconds=env_step_elapsed,
                reward=reward,
                done=done,
                replay_size=replay_buffer.size,
                action_dim=action_dim,
                state=state,
                device=device,
            )

            replay_add_start = time.perf_counter()
            replay_buffer.add(state, node_action, next_state, reward, done)
            replay_add_elapsed = timed_phase(phase_totals, phase_counts, "replay_add", replay_add_start)
            write_phase_row(
                writer=writer,
                args=args,
                step=step,
                episode_step=episode_step,
                phase="replay_add",
                elapsed_seconds=replay_add_elapsed,
                reward=reward,
                done=done,
                replay_size=replay_buffer.size,
                action_dim=action_dim,
                state=state,
                device=device,
            )

            guard_start = time.perf_counter()
            should_train = timestep >= args.start_timesteps
            guard_elapsed = timed_phase(
                phase_totals,
                phase_counts,
                "replay_sample_or_train_guard",
                guard_start,
            )
            write_phase_row(
                writer=writer,
                args=args,
                step=step,
                episode_step=episode_step,
                phase="replay_sample_or_train_guard",
                elapsed_seconds=guard_elapsed,
                reward=reward,
                done=done,
                replay_size=replay_buffer.size,
                action_dim=action_dim,
                state=state,
                device=device,
                policy_train_called=should_train,
            )

            if should_train:
                train_start = time.perf_counter()
                critic_loss, actor_loss = policy.train(replay_buffer, args.batch_size)
                train_elapsed = timed_phase(phase_totals, phase_counts, "policy_train", train_start)
                write_phase_row(
                    writer=writer,
                    args=args,
                    step=step,
                    episode_step=episode_step,
                    phase="policy_train",
                    elapsed_seconds=train_elapsed,
                    reward=reward,
                    done=done,
                    replay_size=replay_buffer.size,
                    action_dim=action_dim,
                    state=state,
                    device=device,
                    policy_train_called=True,
                    critic_loss=critic_loss,
                    actor_loss=actor_loss,
                )

            state = next_state
            if done:
                episode_index += 1
                reset_start = time.perf_counter()
                state, _ = reset_env_state(env, seed=args.seed + episode_index)
                reset_elapsed = timed_phase(phase_totals, phase_counts, "env_reset", reset_start)
                write_phase_row(
                    writer=writer,
                    args=args,
                    step=step,
                    episode_step=episode_step,
                    phase="env_reset",
                    elapsed_seconds=reset_elapsed,
                    reward=reward,
                    done=done,
                    replay_size=replay_buffer.size,
                    action_dim=action_dim,
                    state=state,
                    device=device,
                )
                episode_step = 0

            loop_elapsed = timed_phase(phase_totals, phase_counts, "total_loop_step", loop_start)
            write_phase_row(
                writer=writer,
                args=args,
                step=step,
                episode_step=episode_step,
                phase="total_loop_step",
                elapsed_seconds=loop_elapsed,
                reward=reward,
                done=done,
                replay_size=replay_buffer.size,
                action_dim=action_dim,
                state=state,
                device=device,
            )

            if args.log_every > 0 and step % args.log_every == 0:
                print(
                    f"[{args.run_name}] step={step}/{args.profile_steps} "
                    f"replay_size={replay_buffer.size} "
                    f"policy_train_calls={phase_counts.get('policy_train', 0)}"
                )
            csv_file.flush()

    total_wall_time_seconds = time.perf_counter() - total_wall_start
    summary = build_summary(
        args=args,
        action_dim=action_dim,
        device=device,
        total_wall_time_seconds=total_wall_time_seconds,
        phase_totals=phase_totals,
        phase_counts=phase_counts,
        replay_buffer=replay_buffer,
    )
    with summary_path.open("w") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)
        summary_file.write("\n")

    print("---------------------------------------")
    print("TD3 EV-GNN runtime profile complete")
    print(f"Run name: {args.run_name}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Config: {args.config}")
    print(f"Steps: {args.profile_steps}")
    print(f"Env step total: {summary['total_env_step_time_seconds']:.3f}s")
    print(f"Select action total: {summary['total_select_action_time_seconds']:.3f}s")
    print(f"Policy train total: {summary['total_policy_train_time_seconds']:.3f}s")
    print(f"Replay add total: {summary['total_replay_add_time_seconds']:.3f}s")
    print(f"Policy train calls: {summary['number_of_policy_train_calls']}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {summary_path}")
    print("---------------------------------------")


if __name__ == "__main__":
    main()
