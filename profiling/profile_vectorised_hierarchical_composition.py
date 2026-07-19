#!/usr/bin/env python3
"""Validate and profile reference vs vectorised hierarchical composition.

This harness compares the historical reference composition path against the
Approach A vectorised production path on fixed real PublicPST graph states. It
does not run formal training, does not step optimisers, and does not change
policy parameters.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import socket
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


IMPLEMENTATIONS = ("reference", "vectorised")
TIMED_PHASES = (
    "composition_forward",
    "complete_actor_forward",
    "complete_actor_forward_backward_sum_loss",
    "complete_actor_forward_backward_critic_loss",
)
WORKLOADS = ("online_representative_state", "replay_batch")
SCALES = {
    "25CP": "config_files/PublicPST_25cp.yaml",
    "100CP": "config_files/PublicPST_100.yaml",
    "500CP": "config_files/PublicPST_500.yaml",
}
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_BATCH_SIZES = (1, 8, 16, 32, 64)
FORWARD_ATOL = 1e-6
FORWARD_RTOL = 1e-5
GRADIENT_ATOL = 5e-5
GRADIENT_RTOL = 1e-4
VALIDATION_POLICY = {
    "forward_output_contract": {
        "mode": "hard_fail",
        "description": "Forward parity, output shape/dtype/device, finite values, non-EV zero rows, and action bounds are required.",
    },
    "sum_loss_gradient_audit": {
        "mode": "diagnostic_only",
        "description": "Artificial sum(actor_output) actor-gradient parity is recorded with strict tolerance pass/fail and violation counts but does not stop profiling.",
    },
    "critic_coupled_gradient_audit": {
        "mode": "hard_fail",
        "description": "Critic-coupled TD3 actor-loss gradient parity is the behavioural training-relevant gradient gate.",
    },
}

CSV_FIELDNAMES = [
    "run_name",
    "replication",
    "scale",
    "config_path",
    "seed",
    "workload_kind",
    "load_level",
    "batch_size",
    "implementation",
    "phase",
    "schedule_index",
    "implementation_order_position",
    "status",
    "elapsed_seconds",
    "total_nodes",
    "active_ev_count",
    "charger_count",
    "transformer_count",
    "num_graphs",
    "cpu_thread_count",
    "device",
    "python_version",
    "torch_version",
    "pyg_version",
]


@dataclass
class TimingRecord:
    run_name: str
    replication: int
    scale: str
    config_path: str
    seed: int
    workload_kind: str
    load_level: str
    batch_size: int
    implementation: str
    phase: str
    schedule_index: int
    implementation_order_position: int
    status: str
    elapsed_seconds: float
    total_nodes: int
    active_ev_count: int
    charger_count: int
    transformer_count: int
    num_graphs: int
    cpu_thread_count: int
    device: str
    python_version: str
    torch_version: str
    pyg_version: str

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


def allowed_seed(value: str) -> int:
    parsed_value = int(value)
    if parsed_value not in DEFAULT_SEEDS:
        raise argparse.ArgumentTypeError("seed must be one of 0, 1, 2, 3, 4")
    return parsed_value


def allowed_batch_size(value: str) -> int:
    parsed_value = int(value)
    if parsed_value not in DEFAULT_BATCH_SIZES:
        raise argparse.ArgumentTypeError("batch size must be one of 1, 8, 16, 32, 64")
    return parsed_value


def discover_25cp_config() -> str:
    canonical_path = REPO_ROOT / SCALES["25CP"]
    if canonical_path.exists():
        return SCALES["25CP"]
    candidates = sorted((REPO_ROOT / "config_files").glob("PublicPST*25*cp*.yaml"))
    if not candidates:
        candidates = sorted((REPO_ROOT / "config_files").glob("PublicPST*25*.yaml"))
    if not candidates:
        raise FileNotFoundError("Could not discover a PublicPST 25CP config.")
    return str(candidates[0].relative_to(REPO_ROOT))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile reference vs vectorised hierarchical action composition."
    )
    parser.add_argument("--output_dir", "--output-dir", required=True)
    parser.add_argument("--run_name", "--run-name", default="vectorised_hierarchical_composition")
    parser.add_argument(
        "--replication",
        type=nonnegative_int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--scales", nargs="+", choices=tuple(SCALES), default=list(SCALES))
    parser.add_argument("--seeds", nargs="+", type=allowed_seed, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=WORKLOADS,
        default=list(WORKLOADS),
    )
    parser.add_argument(
        "--batch_sizes",
        "--batch-sizes",
        nargs="+",
        type=allowed_batch_size,
        default=list(DEFAULT_BATCH_SIZES),
    )
    parser.add_argument("--warmup_repetitions", "--warmup-repetitions", type=nonnegative_int, default=5)
    parser.add_argument("--measured_repetitions", "--measured-repetitions", type=positive_int, default=30)
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
    scale_configs = dict(SCALES)
    scale_configs["25CP"] = discover_25cp_config()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.scale_configs = scale_configs
    args.scales = list(dict.fromkeys(args.scales))
    args.seeds = list(dict.fromkeys(args.seeds))
    args.workloads = list(dict.fromkeys(args.workloads))
    args.batch_sizes = list(dict.fromkeys(args.batch_sizes))
    if args.replay_buffer_size < max(args.batch_sizes):
        parser.error("--replay_buffer_size must be at least the largest requested batch size")
    for scale in args.scales:
        config_path = REPO_ROOT / args.scale_configs[scale]
        if not config_path.exists():
            parser.error(f"Missing config for {scale}: {args.scale_configs[scale]}")
    return args


def configure_writable_runtime_caches(output_dir: str | Path) -> None:
    cache_root = Path(output_dir) / ".runtime_cache"
    matplotlib_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as manifest_file:
        for chunk in iter(lambda: manifest_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_provenance_metadata() -> dict[str, str]:
    source_manifest_path = os.environ.get("EV_GNN_SOURCE_MANIFEST", "")
    metadata = {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "source_provenance": os.environ.get("EV_GNN_SOURCE_PROVENANCE", ""),
        "source_manifest_path": source_manifest_path,
    }
    if source_manifest_path:
        manifest_path = Path(source_manifest_path)
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"EV_GNN_SOURCE_MANIFEST does not point to a readable file: {source_manifest_path}"
            )
        metadata["source_manifest_sha256"] = sha256_file(manifest_path)
    return metadata


def synchronise_device(device: str) -> None:
    if device == "cuda":
        import torch

        torch.cuda.synchronize()


def time_callable(callable_object: Callable[[], Any], device: str) -> tuple[Any, float]:
    synchronise_device(device)
    start_time = time.perf_counter()
    result = callable_object()
    synchronise_device(device)
    return result, time.perf_counter() - start_time


def graph_counts(state) -> dict[str, int]:
    sample_lengths = [int(length) for length in getattr(state, "sample_node_length", [])]
    total_nodes = int(sum(sample_lengths)) if sample_lengths else int(len(getattr(state, "node_types", [])))
    return {
        "total_nodes": total_nodes,
        "active_ev_count": int(len(getattr(state, "ev_indexes", []))),
        "charger_count": int(len(getattr(state, "cs_indexes", []))),
        "transformer_count": int(len(getattr(state, "tr_indexes", []))),
        "num_graphs": int(len(sample_lengths) if sample_lengths else 1),
    }


def make_record(
    args: argparse.Namespace,
    scale: str,
    config_path: str,
    seed: int,
    workload_kind: str,
    load_level: str,
    batch_size: int,
    implementation: str,
    phase: str,
    schedule_index: int,
    implementation_order_position: int,
    status: str,
    elapsed_seconds: float,
    state,
    device: str,
) -> TimingRecord:
    import torch
    import torch_geometric

    counts = graph_counts(state)
    return TimingRecord(
        run_name=args.run_name,
        replication=int(args.replication),
        scale=scale,
        config_path=config_path,
        seed=int(seed),
        workload_kind=workload_kind,
        load_level=load_level,
        batch_size=int(batch_size),
        implementation=implementation,
        phase=phase,
        schedule_index=int(schedule_index),
        implementation_order_position=int(implementation_order_position),
        status=status,
        elapsed_seconds=float(elapsed_seconds),
        total_nodes=counts["total_nodes"],
        active_ev_count=counts["active_ev_count"],
        charger_count=counts["charger_count"],
        transformer_count=counts["transformer_count"],
        num_graphs=counts["num_graphs"],
        cpu_thread_count=int(torch.get_num_threads()),
        device=device,
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        pyg_version=torch_geometric.__version__,
    )


def calculate_summary_statistics(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "total": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "standard_deviation": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
            "p95": 0.0,
        }

    sorted_values = sorted(float(value) for value in values)
    p95_index = max(0, math.ceil(0.95 * len(sorted_values)) - 1)
    return {
        "count": int(len(sorted_values)),
        "total": float(sum(sorted_values)),
        "mean": float(statistics.mean(sorted_values)),
        "median": float(statistics.median(sorted_values)),
        "standard_deviation": float(statistics.stdev(sorted_values)) if len(sorted_values) > 1 else 0.0,
        "minimum": float(sorted_values[0]),
        "maximum": float(sorted_values[-1]),
        "p95": float(sorted_values[p95_index]),
    }


def confidence_interval_95(values: list[float]) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    mean_value = statistics.mean(values)
    half_width = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return [float(mean_value - half_width), float(mean_value + half_width)]


def grouped_key(record: TimingRecord, include_implementation: bool) -> tuple[Any, ...]:
    key = (
        record.replication,
        record.scale,
        record.seed,
        record.workload_kind,
        record.load_level,
        record.batch_size,
    )
    if include_implementation:
        key = key + (record.implementation,)
    return key + (record.phase,)


def key_to_dict(key: tuple[Any, ...], include_implementation: bool) -> dict[str, Any]:
    names = [
        "replication",
        "scale",
        "seed",
        "workload_kind",
        "load_level",
        "batch_size",
    ]
    if include_implementation:
        names.append("implementation")
    names.append("phase")
    return dict(zip(names, key))


def build_json_summary(
    records: list[TimingRecord],
    metadata: dict[str, Any],
    numerical_equivalence_audit: list[dict[str, Any]],
) -> dict[str, Any]:
    measured_records = [record for record in records if record.status == "measured"]
    grouped_values: dict[tuple[Any, ...], list[float]] = {}
    for record in measured_records:
        grouped_values.setdefault(grouped_key(record, include_implementation=True), []).append(
            float(record.elapsed_seconds)
        )

    groups = [
        {
            **key_to_dict(key, include_implementation=True),
            **calculate_summary_statistics(values),
        }
        for key, values in sorted(grouped_values.items())
    ]

    paired_speedups = []
    pairable_keys = sorted(
        {
            grouped_key(record, include_implementation=False)
            for record in measured_records
            if record.implementation in IMPLEMENTATIONS
        }
    )
    for pair_key in pairable_keys:
        reference_rows = [
            record
            for record in measured_records
            if record.implementation == "reference"
            and grouped_key(record, include_implementation=False) == pair_key
        ]
        vectorised_rows = [
            record
            for record in measured_records
            if record.implementation == "vectorised"
            and grouped_key(record, include_implementation=False) == pair_key
        ]
        reference_by_schedule = {record.schedule_index: record for record in reference_rows}
        vectorised_by_schedule = {record.schedule_index: record for record in vectorised_rows}
        paired_schedule_indexes = sorted(set(reference_by_schedule) & set(vectorised_by_schedule))
        if not paired_schedule_indexes:
            continue

        reference_values = [float(reference_by_schedule[index].elapsed_seconds) for index in paired_schedule_indexes]
        vectorised_values = [float(vectorised_by_schedule[index].elapsed_seconds) for index in paired_schedule_indexes]
        paired_values = [
            reference_value / vectorised_value
            for reference_value, vectorised_value in zip(reference_values, vectorised_values)
            if vectorised_value > 0.0
        ]
        if not paired_values:
            continue

        paired_speedups.append(
            {
                **key_to_dict(pair_key, include_implementation=False),
                "median_speedup": float(statistics.median(reference_values) / statistics.median(vectorised_values)),
                "mean_speedup": float(statistics.mean(reference_values) / statistics.mean(vectorised_values)),
                "paired_speedup_values": paired_values,
                "paired_speedup_mean": float(statistics.mean(paired_values)),
                "paired_speedup_median": float(statistics.median(paired_values)),
                "paired_speedup_standard_deviation": (
                    float(statistics.stdev(paired_values)) if len(paired_values) > 1 else 0.0
                ),
                "within_run_paired_timing_speedup_95_percent_confidence_interval": confidence_interval_95(
                    paired_values
                ),
                "confidence_interval_method": (
                    "Normal approximation over paired timing repetitions within one profiler run: "
                    "mean +/- 1.96 * sample_standard_deviation / sqrt(n). This interval "
                    "characterises within-run timing variation only and does not represent "
                    "uncertainty across seeds, M3 allocations, nodes, or training runs."
                ),
            }
        )

    residual_bottleneck_ratios = build_residual_bottleneck_ratios(groups)
    return {
        "metadata": metadata,
        "groups": groups,
        "paired_speedups": paired_speedups,
        "numerical_equivalence_audit": numerical_equivalence_audit,
        "validation_policy": VALIDATION_POLICY,
        "residual_bottleneck_ratios": residual_bottleneck_ratios,
        "residual_bottleneck_note": (
            "Ratios are approximate diagnostics because separately timed phases "
            "are not strictly additive."
        ),
    }


def build_residual_bottleneck_ratios(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_by_key = {
        (
            group["replication"],
            group["scale"],
            group["seed"],
            group["workload_kind"],
            group["load_level"],
            group["batch_size"],
            group["implementation"],
            group["phase"],
        ): group
        for group in groups
    }
    ratio_entries = []
    base_keys = sorted(
        {
            key[:6]
            for key in group_by_key
            if key[7] in {"composition_forward", "complete_actor_forward"}
        }
    )
    for base_key in base_keys:
        for implementation in IMPLEMENTATIONS:
            composition = group_by_key.get(base_key + (implementation, "composition_forward"))
            complete = group_by_key.get(base_key + (implementation, "complete_actor_forward"))
            if not composition or not complete or complete["median"] == 0.0:
                continue
            ratio_entries.append(
                {
                    "replication": base_key[0],
                    "scale": base_key[1],
                    "seed": base_key[2],
                    "workload_kind": base_key[3],
                    "load_level": base_key[4],
                    "batch_size": base_key[5],
                    "implementation": implementation,
                    "approximate_composition_forward_to_complete_forward_median_ratio": float(
                        composition["median"] / complete["median"]
                    ),
                }
            )
    return ratio_entries


def write_outputs(
    records: list[TimingRecord],
    summary: dict[str, Any],
    output_dir: str | Path,
    run_name: str,
    replication: int,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / f"{run_name}_rep{replication}_vectorised_hierarchical_composition_steps.csv"
    json_path = output_path / f"{run_name}_rep{replication}_vectorised_hierarchical_composition_summary.json"

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_csv_row())

    with json_path.open("w") as json_file:
        json.dump(summary, json_file, indent=2, sort_keys=True)
        json_file.write("\n")

    return csv_path, json_path


def assert_close(actual, expected) -> None:
    import torch

    torch.testing.assert_close(actual, expected, atol=FORWARD_ATOL, rtol=FORWARD_RTOL)


def tensor_difference_metrics(actual, expected) -> dict[str, float]:
    import torch

    actual = actual.detach()
    expected = expected.detach()
    absolute_difference = (actual - expected).abs()
    if absolute_difference.numel() == 0:
        return {
            "maximum_absolute_difference": 0.0,
            "maximum_relative_difference": 0.0,
        }
    denominator = expected.abs().clamp_min(torch.finfo(expected.dtype).eps)
    relative_difference = absolute_difference / denominator
    return {
        "maximum_absolute_difference": float(absolute_difference.max().detach().cpu()),
        "maximum_relative_difference": float(relative_difference.max().detach().cpu()),
    }


def output_difference_audit(reference_output, vectorised_output) -> dict[str, float]:
    metrics = tensor_difference_metrics(vectorised_output, reference_output)
    return {
        "maximum_absolute_output_difference": metrics["maximum_absolute_difference"],
        "maximum_relative_output_difference": metrics["maximum_relative_difference"],
    }


def loss_difference_audit(reference_loss, vectorised_loss) -> dict[str, float]:
    reference_loss_value = float(reference_loss.detach().cpu())
    vectorised_loss_value = float(vectorised_loss.detach().cpu())
    absolute_difference = abs(vectorised_loss_value - reference_loss_value)
    denominator = max(abs(reference_loss_value), sys.float_info.epsilon)
    return {
        "reference_loss": reference_loss_value,
        "vectorised_loss": vectorised_loss_value,
        "absolute_loss_difference": float(absolute_difference),
        "relative_loss_difference": float(absolute_difference / denominator),
    }


def assert_gradient_close(actual, expected, parameter_name: str) -> None:
    import torch

    try:
        torch.testing.assert_close(
            actual,
            expected,
            atol=GRADIENT_ATOL,
            rtol=GRADIENT_RTOL,
        )
    except AssertionError as close_error:
        raise AssertionError(f"Gradient mismatch for {parameter_name}") from close_error


def gradient_difference_metrics(actual, expected) -> dict[str, float]:
    return tensor_difference_metrics(actual, expected)


def gradient_strict_tolerance_violation_count(actual, expected) -> int:
    import torch

    close_mask = torch.isclose(
        actual.detach(),
        expected.detach(),
        atol=GRADIENT_ATOL,
        rtol=GRADIENT_RTOL,
    )
    return int((~close_mask).sum().detach().cpu())


def assert_non_ev_action_rows_zero(full_node_action, state) -> None:
    import torch

    total_nodes = graph_counts(state)["total_nodes"]
    active_ev_node_indexes = torch.as_tensor(
        getattr(state, "ev_indexes", []),
        dtype=torch.long,
        device=full_node_action.device,
    ).reshape(-1)
    non_ev_mask = torch.ones(total_nodes, dtype=torch.bool, device=full_node_action.device)
    if active_ev_node_indexes.numel() > 0:
        non_ev_mask[active_ev_node_indexes] = False
    non_ev_actions = full_node_action.reshape(-1, 1)[non_ev_mask]
    if not torch.equal(non_ev_actions, torch.zeros_like(non_ev_actions)):
        raise AssertionError("Non-EV action rows must remain zero.")


def assert_output_contract(reference_output, vectorised_output, state, max_action: float) -> None:
    import torch

    if reference_output.shape != vectorised_output.shape:
        raise AssertionError(f"Shape mismatch: {reference_output.shape} != {vectorised_output.shape}")
    if reference_output.dtype != vectorised_output.dtype:
        raise AssertionError(f"Dtype mismatch: {reference_output.dtype} != {vectorised_output.dtype}")
    if reference_output.device != vectorised_output.device:
        raise AssertionError(f"Device mismatch: {reference_output.device} != {vectorised_output.device}")
    assert_close(vectorised_output, reference_output)
    assert_non_ev_action_rows_zero(reference_output, state)
    assert_non_ev_action_rows_zero(vectorised_output, state)
    if not torch.isfinite(reference_output).all() or not torch.isfinite(vectorised_output).all():
        raise AssertionError("Actor outputs must be finite.")
    if not torch.all(vectorised_output >= 0.0):
        raise AssertionError("Vectorised output violated lower action bound.")
    if not torch.all(vectorised_output <= float(max_action)):
        raise AssertionError("Vectorised output violated upper action bound.")


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
            f"Invalid gradients: missing={missing_gradients}, nonfinite={nonfinite_gradients}"
        )


def assert_actor_gradients_close(
    reference_actor,
    vectorised_actor,
    *,
    hard_fail: bool,
    audit_mode: str,
) -> dict[str, Any]:
    reference_parameters = dict(reference_actor.named_parameters())
    vectorised_parameters = dict(vectorised_actor.named_parameters())
    if set(reference_parameters) != set(vectorised_parameters):
        raise AssertionError("Actor parameter sets differ.")
    maximum_absolute_difference = 0.0
    maximum_relative_difference = 0.0
    parameter_with_maximum_absolute_difference = ""
    parameter_with_maximum_relative_difference = ""
    strict_tolerance_passed = True
    violating_parameter_names = []
    violating_element_count_by_parameter = {}
    for parameter_name, reference_parameter in reference_parameters.items():
        vectorised_parameter = vectorised_parameters[parameter_name]
        if reference_parameter.grad is None or vectorised_parameter.grad is None:
            raise AssertionError(f"Missing actor gradient for {parameter_name}")
        if not reference_parameter.grad.isfinite().all() or not vectorised_parameter.grad.isfinite().all():
            raise AssertionError(f"Non-finite actor gradient for {parameter_name}")
        metrics = gradient_difference_metrics(
            vectorised_parameter.grad,
            reference_parameter.grad,
        )
        if metrics["maximum_absolute_difference"] >= maximum_absolute_difference:
            maximum_absolute_difference = metrics["maximum_absolute_difference"]
            parameter_with_maximum_absolute_difference = parameter_name
        if metrics["maximum_relative_difference"] >= maximum_relative_difference:
            maximum_relative_difference = metrics["maximum_relative_difference"]
            parameter_with_maximum_relative_difference = parameter_name
        try:
            assert_gradient_close(
                vectorised_parameter.grad,
                reference_parameter.grad,
                parameter_name,
            )
        except AssertionError:
            strict_tolerance_passed = False
            violating_parameter_names.append(parameter_name)
            violating_element_count_by_parameter[parameter_name] = (
                gradient_strict_tolerance_violation_count(
                    vectorised_parameter.grad,
                    reference_parameter.grad,
                )
            )
            if hard_fail:
                raise
    return {
        "audit_mode": audit_mode,
        "strict_tolerance": {"atol": GRADIENT_ATOL, "rtol": GRADIENT_RTOL},
        "strict_tolerance_passed": strict_tolerance_passed,
        "maximum_absolute_gradient_difference": maximum_absolute_difference,
        "maximum_relative_gradient_difference": maximum_relative_difference,
        "parameter_with_maximum_absolute_difference": parameter_with_maximum_absolute_difference,
        "parameter_with_maximum_relative_difference": parameter_with_maximum_relative_difference,
        "violating_parameter_names": violating_parameter_names,
        "violating_element_count_by_parameter": violating_element_count_by_parameter,
        "total_violating_element_count": int(
            sum(violating_element_count_by_parameter.values())
        ),
    }


def state_for_critic(state, device: str):
    import torch

    from utils.replay_buffer_actiongnn import _batch_graphs

    if isinstance(getattr(state, "ev_features", None), torch.Tensor):
        return state
    return _batch_graphs([state], device=torch.device(device))


def validate_workload(
    actor,
    critic,
    state,
    max_action: float,
    device: str,
    scale: str,
    seed: int,
    workload_kind: str,
    load_level: str,
    batch_size: int,
) -> dict[str, Any]:
    audit_record: dict[str, Any] = {
        "scale": scale,
        "seed": int(seed),
        "workload_kind": workload_kind,
        "load_level": load_level,
        "batch_size": int(batch_size),
    }

    reference_actor = copy.deepcopy(actor)
    vectorised_actor = copy.deepcopy(actor)
    with __import__("torch").no_grad():
        reference_output = reference_actor._forward_reference(state, return_details=False)
        vectorised_output = vectorised_actor._forward_vectorised(state)
    assert_output_contract(reference_output, vectorised_output, state, max_action)
    audit_record["forward_audit"] = output_difference_audit(
        reference_output,
        vectorised_output,
    )

    reference_actor = copy.deepcopy(actor)
    vectorised_actor = copy.deepcopy(actor)
    reference_actor.zero_grad(set_to_none=True)
    vectorised_actor.zero_grad(set_to_none=True)
    reference_sum_output = reference_actor._forward_reference(state, return_details=False)
    vectorised_sum_output = vectorised_actor._forward_vectorised(state)
    reference_sum_loss = reference_sum_output.sum()
    vectorised_sum_loss = vectorised_sum_output.sum()
    reference_sum_loss.backward()
    vectorised_sum_loss.backward()
    assert_close(vectorised_sum_output.detach(), reference_sum_output.detach())
    assert_close(vectorised_sum_loss.detach(), reference_sum_loss.detach())
    assert_finite_gradients(reference_actor)
    assert_finite_gradients(vectorised_actor)
    audit_record["sum_loss_gradient_audit"] = assert_actor_gradients_close(
        reference_actor,
        vectorised_actor,
        hard_fail=False,
        audit_mode="diagnostic_only",
    )

    critic_state = state_for_critic(state, device)
    reference_actor = copy.deepcopy(actor)
    vectorised_actor = copy.deepcopy(actor)
    reference_critic = copy.deepcopy(critic)
    vectorised_critic = copy.deepcopy(critic)
    reference_actor.zero_grad(set_to_none=True)
    vectorised_actor.zero_grad(set_to_none=True)
    reference_critic.zero_grad(set_to_none=True)
    vectorised_critic.zero_grad(set_to_none=True)
    reference_critic_output = reference_actor._forward_reference(critic_state, return_details=False)
    vectorised_critic_output = vectorised_actor._forward_vectorised(critic_state)
    reference_critic_loss = -reference_critic.Q1(critic_state, reference_critic_output).mean()
    vectorised_critic_loss = -vectorised_critic.Q1(critic_state, vectorised_critic_output).mean()
    reference_critic_loss.backward()
    vectorised_critic_loss.backward()
    assert_close(vectorised_critic_output.detach(), reference_critic_output.detach())
    assert_close(vectorised_critic_loss.detach(), reference_critic_loss.detach())
    assert_finite_gradients(reference_actor)
    assert_finite_gradients(vectorised_actor)
    audit_record["critic_coupled_gradient_audit"] = {
        **assert_actor_gradients_close(
            reference_actor,
            vectorised_actor,
            hard_fail=True,
            audit_mode="hard_fail",
        ),
        **loss_difference_audit(reference_critic_loss, vectorised_critic_loss),
    }
    return audit_record


def implementation_order_for_schedule_index(schedule_index: int) -> tuple[str, str]:
    if int(schedule_index) % 2 == 0:
        return ("reference", "vectorised")
    return ("vectorised", "reference")


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


def reset_env_state(env, seed: int):
    from utils.ev2gym_training_utils import reset_env

    reset_result = reset_env(env, seed=seed)
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        return reset_result
    return reset_result, {}


def zero_full_node_action_for_state(state):
    import torch

    return torch.zeros((graph_counts(state)["total_nodes"], 1), dtype=torch.float32)


def select_representative_single_state(states: list[Any]) -> Any:
    if not states:
        raise ValueError("At least one candidate state is required.")
    representative_state = states[0]
    representative_rank = (
        graph_counts(representative_state)["active_ev_count"],
        graph_counts(representative_state)["total_nodes"],
    )
    for candidate_state in states[1:]:
        candidate_rank = (
            graph_counts(candidate_state)["active_ev_count"],
            graph_counts(candidate_state)["total_nodes"],
        )
        if candidate_rank > representative_rank:
            representative_state = candidate_state
            representative_rank = candidate_rank
    return representative_state


def select_load_coverage_states(active_states: list[Any]) -> dict[str, Any]:
    if not active_states:
        raise RuntimeError("No active states available for load coverage selection.")
    count_by_state = [(graph_counts(state)["active_ev_count"], order, state) for order, state in enumerate(active_states)]
    low_count = min(count for count, _, _ in count_by_state if count > 0)
    high_count = max(count for count, _, _ in count_by_state)
    midpoint = (low_count + high_count) / 2.0

    low_state = min((entry for entry in count_by_state if entry[0] == low_count), key=lambda entry: entry[1])[2]
    high_state = min((entry for entry in count_by_state if entry[0] == high_count), key=lambda entry: entry[1])[2]
    medium_state = min(count_by_state, key=lambda entry: (abs(entry[0] - midpoint), entry[1]))[2]
    return {
        "low_active_ev_state": low_state,
        "medium_active_ev_state": medium_state,
        "high_active_ev_state": high_state,
    }


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
    active_states = []
    replay_threshold_reached_at_step = None
    replay_threshold_episode_index = None
    completed_collection_episodes = 0
    collected_active_transition_count = 0
    episode_index = 0

    for collection_step in range(1, int(max_collection_steps) + 1):
        state_has_active_ev = len(getattr(state, "ev_indexes", [])) > 0
        if state_has_active_ev:
            active_states.append(state)

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

        if replay_threshold_reached_at_step is not None and done:
            representative_state = select_representative_single_state(active_states)
            load_states = select_load_coverage_states(active_states)
            return {
                "representative_single_state": representative_state,
                "representative_single_state_counts": graph_counts(representative_state),
                "load_coverage_states": load_states,
                "load_coverage_state_counts": {
                    load_level: graph_counts(load_state)
                    for load_level, load_state in load_states.items()
                },
                "collection_steps_completed": collection_step,
                "collected_active_transition_count": collected_active_transition_count,
                "completed_collection_episodes": completed_collection_episodes,
                "replay_threshold_reached_at_step": replay_threshold_reached_at_step,
                "replay_threshold_episode_index": replay_threshold_episode_index,
                "collection_termination_reason": "completed_threshold_episode",
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
        raise RuntimeError("Replay threshold episode did not terminate before max_collection_steps.")
    raise RuntimeError(
        "Could not collect enough active-EV transitions. "
        f"Collected {collected_active_transition_count}, required {target_replay_size}."
    )


def collect_public_pst_replay_data(args: argparse.Namespace, config_path: str, seed: int, device: str):
    import numpy as np

    from utils.ev2gym_training_utils import make_env, normalise_step_result, set_global_seed
    from utils.replay_buffer_actiongnn import ActionGNN_ReplayBuffer

    set_global_seed(seed)
    env = make_env(config_path, seed=seed)
    action_dim = int(env.action_space.shape[0])
    max_action = float(env.action_space.high[0])
    replay_buffer = ActionGNN_ReplayBuffer(
        action_dim=action_dim,
        max_size=args.replay_buffer_size,
        device=device,
    )

    zero_mapped_action = np.zeros(action_dim, dtype=np.float32)
    target_replay_size = max(args.batch_sizes)
    initial_state, _ = reset_env_state(env, seed=seed)

    def step_state(state):
        return normalise_step_result(env.step(zero_mapped_action))

    def reset_collection_episode(episode_index):
        reset_state, _ = reset_env_state(env, seed=seed + episode_index)
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


def fixed_replay_batches(seed: int, batch_sizes: list[int], replay_buffer) -> dict[int, Any]:
    import numpy as np

    batches = {}
    for batch_size in batch_sizes:
        np.random.seed(seed + 10_000 + batch_size)
        batches[batch_size] = replay_buffer.sample(batch_size)
    return batches


def make_actor_and_critic(args: argparse.Namespace, max_action: float, device: str):
    from TD3.TD3_HierarchicalActionGNN import Actor, Critic
    from utils.state_public_pst_gnn import PublicPST_GNN

    actor = Actor(
        max_action=max_action,
        fx_node_sizes=PublicPST_GNN.node_sizes,
        feature_dim=args.fx_dim,
        GNN_hidden_dim=args.fx_GNN_hidden_dim,
        num_gcn_layers=args.actor_num_gcn_layers,
        discrete_actions=1,
        device=__import__("torch").device(device),
    ).to(device)
    critic = Critic(
        fx_node_sizes=PublicPST_GNN.node_sizes,
        feature_dim=args.fx_dim,
        GNN_hidden_dim=args.fx_GNN_hidden_dim,
        mlp_hidden_dim=args.mlp_hidden_dim,
        discrete_actions=1,
        num_gcn_layers=args.critic_num_gcn_layers,
        device=__import__("torch").device(device),
    ).to(device)
    return actor, critic


def prepare_composition_inputs(actor, state):
    with __import__("torch").no_grad():
        (
            _embedded_node_features,
            _edge_index,
            ev_features,
            cs_features,
            tr_features,
            active_ev_node_indexes,
            charger_node_indexes,
            transformer_node_indexes,
            node_embeddings,
        ) = actor._prepare_forward_inputs(state)
    return {
        "state": state,
        "node_embeddings": node_embeddings,
        "ev_features": ev_features,
        "cs_features": cs_features,
        "tr_features": tr_features,
        "active_ev_node_indexes": active_ev_node_indexes,
        "charger_node_indexes": charger_node_indexes,
        "transformer_node_indexes": transformer_node_indexes,
    }


def run_composition_forward_from_inputs(actor, inputs, implementation: str):
    if implementation == "reference":
        return actor._compose_full_node_action_reference(**inputs)[0]
    return actor._compose_full_node_action_vectorised(**inputs)


def run_complete_forward(actor, state, implementation: str):
    if implementation == "reference":
        return actor._forward_reference(state, return_details=False)
    return actor._forward_vectorised(state)


def run_backward_sum_loss(actor, state, implementation: str):
    actor.zero_grad(set_to_none=True)
    actor_output = run_complete_forward(actor, state, implementation)
    loss = actor_output.sum()
    loss.backward()
    return actor_output, loss.detach()


def run_backward_critic_loss(actor, critic, state, implementation: str):
    actor.zero_grad(set_to_none=True)
    critic.zero_grad(set_to_none=True)
    actor_output = run_complete_forward(actor, state, implementation)
    actor_loss = -critic.Q1(state, actor_output).mean()
    actor_loss.backward()
    return actor_output, actor_loss.detach()


def time_phase(actor, critic, state, critic_state, implementation: str, phase: str, device: str):
    import torch

    if phase == "composition_forward":
        with torch.no_grad():
            composition_inputs = prepare_composition_inputs(actor, state)
            return time_callable(
                lambda: run_composition_forward_from_inputs(
                    actor,
                    composition_inputs,
                    implementation,
                ),
                device,
            )
    if phase == "complete_actor_forward":
        with torch.no_grad():
            return time_callable(lambda: run_complete_forward(actor, state, implementation), device)
    if phase == "complete_actor_forward_backward_sum_loss":
        return time_callable(lambda: run_backward_sum_loss(actor, state, implementation), device)
    if phase == "complete_actor_forward_backward_critic_loss":
        return time_callable(lambda: run_backward_critic_loss(actor, critic, critic_state, implementation), device)
    raise ValueError(f"Unsupported phase: {phase}")


def run_workload_measurements(
    args: argparse.Namespace,
    scale: str,
    config_path: str,
    seed: int,
    actor,
    critic,
    workload_state,
    workload_kind: str,
    load_level: str,
    batch_size: int,
    max_action: float,
    device: str,
    numerical_equivalence_audit: list[dict[str, Any]],
) -> list[TimingRecord]:
    numerical_equivalence_audit.append(
        validate_workload(
            actor,
            critic,
            workload_state,
            max_action=max_action,
            device=device,
            scale=scale,
            seed=seed,
            workload_kind=workload_kind,
            load_level=load_level,
            batch_size=batch_size,
        )
    )
    critic_state = state_for_critic(workload_state, device)
    records = []
    for schedule_index, status, _ in repetition_schedule(args):
        for implementation_order_position, implementation in enumerate(
            implementation_order_for_schedule_index(schedule_index)
        ):
            timed_actor = copy.deepcopy(actor)
            timed_critic = copy.deepcopy(critic)
            for phase in TIMED_PHASES:
                _, elapsed_seconds = time_phase(
                    timed_actor,
                    timed_critic,
                    workload_state,
                    critic_state,
                    implementation,
                    phase,
                    device,
                )
                records.append(
                    make_record(
                        args=args,
                        scale=scale,
                        config_path=config_path,
                        seed=seed,
                        workload_kind=workload_kind,
                        load_level=load_level,
                        batch_size=batch_size,
                        implementation=implementation,
                        phase=phase,
                        schedule_index=schedule_index,
                        implementation_order_position=implementation_order_position,
                        status=status,
                        elapsed_seconds=elapsed_seconds,
                        state=workload_state,
                        device=device,
                    )
                )
    return records


def run_replay_sampling_context(
    args: argparse.Namespace,
    scale: str,
    config_path: str,
    seed: int,
    replay_buffer,
    batch_size: int,
    device: str,
) -> list[TimingRecord]:
    import numpy as np

    records = []
    for schedule_index, status, _ in repetition_schedule(args):
        sample_seed = seed + batch_size * 100_000 + schedule_index
        np.random.seed(sample_seed)
        replay_sample, elapsed_seconds = time_callable(
            lambda: replay_buffer.sample(batch_size),
            device,
        )
        records.append(
            make_record(
                args=args,
                scale=scale,
                config_path=config_path,
                seed=seed,
                workload_kind="replay_batch",
                load_level="fixed_replay_sample",
                batch_size=batch_size,
                implementation="context",
                phase="replay_sampling_context",
                schedule_index=schedule_index,
                implementation_order_position=-1,
                status=status,
                elapsed_seconds=elapsed_seconds,
                state=replay_sample[0],
                device=device,
            )
        )
    return records


def run_measurements(args: argparse.Namespace) -> tuple[list[TimingRecord], dict[str, Any], list[dict[str, Any]]]:
    import torch
    import torch_geometric

    from utils.ev2gym_training_utils import resolve_device, set_global_seed

    device = resolve_device(args.device)
    records: list[TimingRecord] = []
    numerical_equivalence_audit: list[dict[str, Any]] = []
    collection_metadata = []
    total_wall_start = time.perf_counter()

    for scale in args.scales:
        config_path = args.scale_configs[scale]
        for seed in args.seeds:
            set_global_seed(seed)
            collected_data = collect_public_pst_replay_data(args, config_path, seed, device)
            replay_buffer = collected_data["replay_buffer"]
            fixed_batches = fixed_replay_batches(seed, args.batch_sizes, replay_buffer)
            actor, critic = make_actor_and_critic(args, collected_data["max_action"], device)

            for load_level, load_state in collected_data["load_coverage_states"].items():
                numerical_equivalence_audit.append(
                    validate_workload(
                        actor,
                        critic,
                        load_state,
                        max_action=collected_data["max_action"],
                        device=device,
                        scale=scale,
                        seed=seed,
                        workload_kind="load_coverage_state",
                        load_level=load_level,
                        batch_size=1,
                    )
                )

            if "online_representative_state" in args.workloads:
                records.extend(
                    run_workload_measurements(
                        args=args,
                        scale=scale,
                        config_path=config_path,
                        seed=seed,
                        actor=actor,
                        critic=critic,
                        workload_state=collected_data["representative_single_state"],
                        workload_kind="online_representative_state",
                        load_level="high_active_ev_state",
                        batch_size=1,
                        max_action=collected_data["max_action"],
                        device=device,
                        numerical_equivalence_audit=numerical_equivalence_audit,
                    )
                )

            if "replay_batch" in args.workloads:
                for batch_size in args.batch_sizes:
                    records.extend(
                        run_replay_sampling_context(
                            args=args,
                            scale=scale,
                            config_path=config_path,
                            seed=seed,
                            replay_buffer=replay_buffer,
                            batch_size=batch_size,
                            device=device,
                        )
                    )
                    records.extend(
                        run_workload_measurements(
                            args=args,
                            scale=scale,
                            config_path=config_path,
                            seed=seed,
                            actor=actor,
                            critic=critic,
                            workload_state=fixed_batches[batch_size][0],
                            workload_kind="replay_batch",
                            load_level="fixed_replay_sample",
                            batch_size=batch_size,
                            max_action=collected_data["max_action"],
                            device=device,
                            numerical_equivalence_audit=numerical_equivalence_audit,
                        )
                    )

            collection_metadata.append(
                {
                    "scale": scale,
                    "config_path": config_path,
                    "seed": seed,
                    "collection_steps_completed": int(collected_data["collection_steps_completed"]),
                    "collected_active_transition_count": int(collected_data["collected_active_transition_count"]),
                    "completed_collection_episodes": int(collected_data["completed_collection_episodes"]),
                    "replay_threshold_reached_at_step": int(collected_data["replay_threshold_reached_at_step"]),
                    "replay_threshold_episode_index": int(collected_data["replay_threshold_episode_index"]),
                    "collection_termination_reason": collected_data["collection_termination_reason"],
                    "representative_single_state_counts": collected_data["representative_single_state_counts"],
                    "load_coverage_state_counts": collected_data["load_coverage_state_counts"],
                }
            )

    metadata = {
        "run_name": args.run_name,
        "replication": int(args.replication),
        "scales": args.scales,
        "scale_configs": {scale: args.scale_configs[scale] for scale in args.scales},
        "seeds": [int(seed) for seed in args.seeds],
        "workloads": args.workloads,
        "batch_sizes": [int(batch_size) for batch_size in args.batch_sizes],
        "warmup_repetitions": int(args.warmup_repetitions),
        "measured_repetitions": int(args.measured_repetitions),
        "timed_phases": list(TIMED_PHASES),
        "default_full_matrix": {
            "scales": list(SCALES),
            "seeds": list(DEFAULT_SEEDS),
            "workloads": list(WORKLOADS),
            "batch_sizes": list(DEFAULT_BATCH_SIZES),
        },
        "load_coverage_levels": [
            "low_active_ev_state",
            "medium_active_ev_state",
            "high_active_ev_state",
        ],
        "paired_confidence_interval_method": (
            "Normal approximation over paired timing repetitions within one profiler run: "
            "mean +/- 1.96 * sample_standard_deviation / sqrt(n). This interval "
            "characterises within-run timing variation only and does not represent "
            "uncertainty across seeds, M3 allocations, nodes, or training runs."
        ),
        "forward_parity_tolerance": {"atol": FORWARD_ATOL, "rtol": FORWARD_RTOL},
        "gradient_parity_tolerance": {"atol": GRADIENT_ATOL, "rtol": GRADIENT_RTOL},
        "device": device,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_geometric_version": torch_geometric.__version__,
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
        "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS", ""),
        "collection_metadata": collection_metadata,
        "total_wall_time_seconds": float(time.perf_counter() - total_wall_start),
        **runtime_provenance_metadata(),
    }
    return records, metadata, numerical_equivalence_audit


def print_numerical_audit_summary(numerical_equivalence_audit: list[dict[str, Any]]) -> None:
    if not numerical_equivalence_audit:
        print("Numerical equivalence audit: no validated workloads recorded")
        return

    max_output_abs = max(
        numerical_equivalence_audit,
        key=lambda record: record["forward_audit"]["maximum_absolute_output_difference"],
    )
    max_output_rel = max(
        numerical_equivalence_audit,
        key=lambda record: record["forward_audit"]["maximum_relative_output_difference"],
    )
    max_sum_grad_abs = max(
        numerical_equivalence_audit,
        key=lambda record: record["sum_loss_gradient_audit"]["maximum_absolute_gradient_difference"],
    )
    max_sum_grad_rel = max(
        numerical_equivalence_audit,
        key=lambda record: record["sum_loss_gradient_audit"]["maximum_relative_gradient_difference"],
    )
    max_critic_grad_abs = max(
        numerical_equivalence_audit,
        key=lambda record: record["critic_coupled_gradient_audit"]["maximum_absolute_gradient_difference"],
    )
    max_critic_grad_rel = max(
        numerical_equivalence_audit,
        key=lambda record: record["critic_coupled_gradient_audit"]["maximum_relative_gradient_difference"],
    )
    max_loss_abs = max(
        numerical_equivalence_audit,
        key=lambda record: record["critic_coupled_gradient_audit"]["absolute_loss_difference"],
    )
    max_loss_rel = max(
        numerical_equivalence_audit,
        key=lambda record: record["critic_coupled_gradient_audit"]["relative_loss_difference"],
    )

    print("Numerical equivalence audit maxima:")
    print(
        "  maximum output absolute difference: "
        f"{max_output_abs['forward_audit']['maximum_absolute_output_difference']:.9g}"
    )
    print(
        "  maximum output relative difference: "
        f"{max_output_rel['forward_audit']['maximum_relative_output_difference']:.9g}"
    )
    print(
        "  maximum critic-loss absolute difference: "
        f"{max_loss_abs['critic_coupled_gradient_audit']['absolute_loss_difference']:.9g}"
    )
    print(
        "  maximum critic-loss relative difference: "
        f"{max_loss_rel['critic_coupled_gradient_audit']['relative_loss_difference']:.9g}"
    )
    print(
        "  maximum sum-loss gradient absolute difference: "
        f"{max_sum_grad_abs['sum_loss_gradient_audit']['maximum_absolute_gradient_difference']:.9g} "
        f"({max_sum_grad_abs['sum_loss_gradient_audit']['parameter_with_maximum_absolute_difference']})"
    )
    print(
        "  maximum sum-loss gradient relative difference: "
        f"{max_sum_grad_rel['sum_loss_gradient_audit']['maximum_relative_gradient_difference']:.9g} "
        f"({max_sum_grad_rel['sum_loss_gradient_audit']['parameter_with_maximum_relative_difference']})"
    )
    print(
        "  maximum critic-loss gradient absolute difference: "
        f"{max_critic_grad_abs['critic_coupled_gradient_audit']['maximum_absolute_gradient_difference']:.9g} "
        f"({max_critic_grad_abs['critic_coupled_gradient_audit']['parameter_with_maximum_absolute_difference']})"
    )
    print(
        "  maximum critic-loss gradient relative difference: "
        f"{max_critic_grad_rel['critic_coupled_gradient_audit']['maximum_relative_gradient_difference']:.9g} "
        f"({max_critic_grad_rel['critic_coupled_gradient_audit']['parameter_with_maximum_relative_difference']})"
    )


def main() -> None:
    args = parse_args()
    configure_writable_runtime_caches(args.output_dir)
    records, metadata, numerical_equivalence_audit = run_measurements(args)
    summary = build_json_summary(records, metadata, numerical_equivalence_audit)
    csv_path, json_path = write_outputs(
        records=records,
        summary=summary,
        output_dir=args.output_dir,
        run_name=args.run_name,
        replication=args.replication,
    )

    print("---------------------------------------")
    print("Vectorised hierarchical composition profile complete")
    print(f"Run name: {args.run_name}")
    print(f"Replication: {args.replication}")
    print(f"Scales: {args.scales}")
    print(f"Seeds: {args.seeds}")
    print(f"Workloads: {args.workloads}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Warm-ups: {args.warmup_repetitions}")
    print(f"Measured repetitions: {args.measured_repetitions}")
    print_numerical_audit_summary(numerical_equivalence_audit)
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print("VECTORISED_HIERARCHICAL_COMPOSITION_PROFILE_COMPLETE")
    print("---------------------------------------")


if __name__ == "__main__":
    main()
