#!/usr/bin/env python3
"""Audit repository readiness for a multi-scale, multi-algorithm matrix.

The audit is intentionally static: it reads local source, config, job, and
documentation files without importing EV2Gym, torch, stable-baselines3, or other
training dependencies.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


EXPECTED_SCALES = ("25CP", "100CP", "500CP", "1000CP")
CONFIG_FIELDS = (
    "timescale",
    "simulation_length",
    "number_of_charging_stations",
    "number_of_transformers",
    "number_of_ports_per_cs",
    "charging_network_topology",
    "transformer.max_power",
    "charging_station.min_charge_current",
    "charging_station.max_charge_current",
    "charging_station.min_discharge_current",
    "charging_station.max_discharge_current",
    "charging_station.voltage",
    "charging_station.phases",
)

KEY_METRICS_TO_AUDIT = (
    "episode_reward",
    "tracking_error",
    "energy_tracking_error",
    "power_tracker_violation",
    "average_user_satisfaction",
    "energy_user_satisfaction",
    "total_transformer_overload",
    "total_ev_served",
    "total_energy_charged",
    "total_energy_discharged",
    "battery_degradation",
    "battery_degradation_calendar",
    "battery_degradation_cycling",
    "total_profits",
    "action_fraction_at_max",
)

ACTION_DIAGNOSTIC_METRICS = (
    "action_mean",
    "action_std",
    "action_min",
    "action_max",
    "action_fraction_zero",
    "action_fraction_at_max",
    "active_action_count_mean",
)

SB3_ALGORITHM_SPECS = (
    {
        "display": "PPO",
        "branch": "ppo",
        "class_name": "PPO",
        "continuous": "yes",
        "requires_discrete_wrapper": "no",
        "special_wrapper": "optional ThreeStep_Action_DiscreteActionSpace only when --discrete_actions is used",
        "adapter_note": "SB3 checkpoint load adapter, deterministic eval30 policy adapter, canonical CSV schema, seed protocol, and wrapper metadata.",
    },
    {
        "display": "A2C",
        "branch": "a2c",
        "class_name": "A2C",
        "continuous": "yes",
        "requires_discrete_wrapper": "no",
        "special_wrapper": "optional ThreeStep_Action_DiscreteActionSpace only when --discrete_actions is used",
        "adapter_note": "SB3 checkpoint load adapter, deterministic eval30 policy adapter, canonical CSV schema, seed protocol, and wrapper metadata.",
    },
    {
        "display": "DDPG",
        "branch": "ddpg",
        "class_name": "DDPG",
        "continuous": "yes",
        "requires_discrete_wrapper": "no",
        "special_wrapper": "none detected for continuous PublicPST use",
        "adapter_note": "SB3 checkpoint load adapter, deterministic eval30 policy adapter, canonical CSV schema, seed protocol, and wrapper metadata.",
    },
    {
        "display": "SAC",
        "branch": "sac",
        "class_name": "SAC",
        "continuous": "yes",
        "requires_discrete_wrapper": "no",
        "special_wrapper": "none detected for continuous PublicPST use",
        "adapter_note": "SB3 checkpoint load adapter, deterministic eval30 policy adapter, canonical CSV schema, seed protocol, and wrapper metadata.",
    },
    {
        "display": "TD3",
        "branch": "td3",
        "class_name": "TD3",
        "continuous": "yes",
        "requires_discrete_wrapper": "no",
        "special_wrapper": "none detected for continuous PublicPST use",
        "adapter_note": "SB3 checkpoint load adapter, deterministic eval30 policy adapter, canonical CSV schema, seed protocol, and wrapper metadata.",
    },
    {
        "display": "DQN",
        "branch": "DQN",
        "class_name": "DQN",
        "continuous": "no",
        "requires_discrete_wrapper": "yes",
        "special_wrapper": "Fully_Discrete is selected when --discrete_actions is true",
        "adapter_note": "Discrete-action wrapper validation plus SB3 checkpoint load adapter, deterministic eval30 policy adapter, canonical CSV schema, seed protocol, and wrapper metadata.",
    },
    {
        "display": "TQC",
        "branch": "tqc",
        "class_name": "TQC",
        "continuous": "yes",
        "requires_discrete_wrapper": "no",
        "special_wrapper": "none detected for continuous PublicPST use",
        "adapter_note": "SB3 checkpoint load adapter, deterministic eval30 policy adapter, canonical CSV schema, seed protocol, and wrapper metadata.",
    },
    {
        "display": "TRPO",
        "branch": "trpo",
        "class_name": "TRPO",
        "continuous": "yes",
        "requires_discrete_wrapper": "no",
        "special_wrapper": "optional ThreeStep_Action_DiscreteActionSpace only when --discrete_actions is used",
        "adapter_note": "SB3 checkpoint load adapter, deterministic eval30 policy adapter, canonical CSV schema, seed protocol, and wrapper metadata.",
    },
    {
        "display": "ARS",
        "branch": "ars",
        "class_name": "ARS",
        "continuous": "yes",
        "requires_discrete_wrapper": "no",
        "special_wrapper": "none detected for continuous PublicPST use",
        "adapter_note": "SB3 checkpoint load adapter, deterministic eval30 policy adapter, canonical CSV schema, seed protocol, and wrapper metadata.",
    },
    {
        "display": "RecurrentPPO",
        "branch": "RecurrentPPO",
        "class_name": "RecurrentPPO",
        "continuous": "yes",
        "requires_discrete_wrapper": "no",
        "special_wrapper": "recurrent state handling is required during deterministic evaluation",
        "adapter_note": "SB3 recurrent-state eval adapter, checkpoint load adapter, canonical CSV schema, seed protocol, and wrapper metadata.",
    },
    {
        "display": "MaskablePPO",
        "branch": "MaskablePPO",
        "class_name": "MaskablePPO",
        "continuous": "no",
        "requires_discrete_wrapper": "yes",
        "special_wrapper": "ActionMasker with mask_fn; discrete action wrapper must be validated",
        "adapter_note": "Action-mask and discrete-wrapper validation plus SB3 checkpoint load adapter, deterministic eval30 adapter, canonical CSV schema, seed protocol, and wrapper metadata.",
    },
    {
        "display": "QRDQN",
        "branch": "QRDQN",
        "class_name": "QRDQN",
        "continuous": "no",
        "requires_discrete_wrapper": "yes",
        "special_wrapper": "Fully_Discrete is selected when --discrete_actions is true",
        "adapter_note": "Discrete-action wrapper validation plus SB3 checkpoint load adapter, deterministic eval30 policy adapter, canonical CSV schema, seed protocol, and wrapper metadata.",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Static audit for EV-GNN multi-scale, multi-algorithm experiment readiness.",
    )
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def relative_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({fieldname: row.get(fieldname, "") for fieldname in fieldnames})


def clean_yaml_value(raw_value: str) -> str:
    return raw_value.split("#", 1)[0].strip()


def parse_simple_yaml_fields(path: Path) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    section_stack: List[Tuple[int, str]] = []

    for raw_line in read_text(path).splitlines():
        stripped_line = raw_line.strip()
        if not stripped_line or stripped_line.startswith("#") or ":" not in stripped_line:
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        while section_stack and indent <= section_stack[-1][0]:
            section_stack.pop()

        key_text, raw_value = stripped_line.split(":", 1)
        key = key_text.strip()
        value = clean_yaml_value(raw_value)
        parent_keys = [section_key for section_indent, section_key in section_stack]
        full_key = ".".join([*parent_keys, key])

        if value:
            fields[full_key] = value
        else:
            fields[full_key] = ""
            section_stack.append((indent, key))

    return fields


def infer_scale_from_filename(path: Path) -> str:
    filename = path.name.lower()
    match = re.search(r"(?:^|_)(25|100|500|1000)(?:cp)?(?:_|\.|$)", filename)
    if not match:
        return ""
    return f"{match.group(1)}CP"


def infer_scale_from_station_count(station_count: str) -> str:
    station_count_text = str(station_count).strip()
    if station_count_text in {"25", "100", "500", "1000"}:
        return f"{station_count_text}CP"
    return ""


def is_public_pst_config(path: Path) -> bool:
    return path.suffix.lower() in {".yaml", ".yml"} and "publicpst" in path.name.lower()


def audit_config_inventory(repo_root: Path) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    config_dir = repo_root / "config_files"
    config_paths = sorted(path for path in config_dir.glob("*.y*ml") if is_public_pst_config(path))
    rows: List[Dict[str, object]] = []
    present_scales = set()
    canonical_paths_by_scale: Dict[str, List[str]] = {scale: [] for scale in EXPECTED_SCALES}

    for config_path in config_paths:
        parsed_fields = parse_simple_yaml_fields(config_path)
        inferred_scale = infer_scale_from_filename(config_path)
        station_scale = infer_scale_from_station_count(parsed_fields.get("number_of_charging_stations", ""))
        if inferred_scale in EXPECTED_SCALES:
            present_scales.add(inferred_scale)
            canonical_paths_by_scale[inferred_scale].append(relative_path(config_path, repo_root))

        required_fields = ("timescale", "simulation_length", "number_of_charging_stations", "number_of_transformers")
        missing_required_fields = [field for field in required_fields if not parsed_fields.get(field)]
        scale_matches_expected = inferred_scale in EXPECTED_SCALES or station_scale in EXPECTED_SCALES
        is_example = "example" in config_path.name.lower()
        appears_usable = "yes" if not missing_required_fields and scale_matches_expected and not is_example else "no"
        notes = []
        if not inferred_scale:
            notes.append("No CP scale inferred from filename.")
        if is_example:
            notes.append("Example config, not a formal matrix scale.")
        if missing_required_fields:
            notes.append(f"Missing required fields: {', '.join(missing_required_fields)}.")
        if inferred_scale and station_scale and inferred_scale != station_scale:
            notes.append(f"Filename scale {inferred_scale} differs from station-count scale {station_scale}.")

        row: Dict[str, object] = {
            "record_type": "config",
            "path": relative_path(config_path, repo_root),
            "exists": "yes",
            "inferred_scale": inferred_scale,
            "scale_from_charging_station_count": station_scale,
            "appears_usable_for_formal_controlled_experiments": appears_usable,
            "missing_required_fields": "; ".join(missing_required_fields),
            "missing_expected_scales": "",
            "notes": " ".join(notes),
        }
        for field in CONFIG_FIELDS:
            row[field] = parsed_fields.get(field, "")
        rows.append(row)

    missing_expected_scales = [scale for scale in EXPECTED_SCALES if scale not in present_scales]
    for scale in EXPECTED_SCALES:
        rows.append({
            "record_type": "expected_scale",
            "path": "; ".join(canonical_paths_by_scale[scale]),
            "exists": "yes" if scale in present_scales else "no",
            "inferred_scale": scale,
            "scale_from_charging_station_count": "",
            "appears_usable_for_formal_controlled_experiments": "yes" if scale in present_scales else "no",
            "missing_required_fields": "",
            "missing_expected_scales": "; ".join(missing_expected_scales) if missing_expected_scales else "none",
            "notes": "Expected scale present." if scale in present_scales else "Expected scale missing.",
        })

    summary = {
        "present_scales": sorted(present_scales, key=EXPECTED_SCALES.index),
        "missing_expected_scales": missing_expected_scales,
        "canonical_paths_by_scale": canonical_paths_by_scale,
        "config_count": len(config_paths),
    }
    return rows, summary


def parse_tuple_string_assignment(text: str, assignment_name: str) -> List[str]:
    pattern = re.compile(rf"{re.escape(assignment_name)}\s*=\s*\((.*?)\)", re.DOTALL)
    match = pattern.search(text)
    if not match:
        return []
    return re.findall(r"['\"]([^'\"]+)['\"]", match.group(1))


def parse_list_or_set_strings(text: str, assignment_name: str) -> List[str]:
    pattern = re.compile(rf"{re.escape(assignment_name)}\s*=\s*[\[\{{](.*?)[\]\}}]", re.DOTALL)
    match = pattern.search(text)
    if not match:
        return []
    return re.findall(r"['\"]([^'\"]+)['\"]", match.group(1))


def branch_is_implemented(text: str, branch_label: str) -> bool:
    quoted_label = re.escape(branch_label)
    return bool(re.search(rf"algorithm\s*==\s*['\"]{quoted_label}['\"]", text))


def class_is_imported(text: str, class_name: str) -> bool:
    return bool(re.search(rf"\b{re.escape(class_name)}\b", text))


def audit_algorithm_inventory(repo_root: Path) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    train_path = repo_root / "train_td3_gnn.py"
    eval_path = repo_root / "evaluate_td3_gnn.py"
    sb3_path = repo_root / "train_baselines.py"
    train_text = read_text(train_path)
    eval_text = read_text(eval_path)
    sb3_text = read_text(sb3_path)

    train_choices = parse_tuple_string_assignment(train_text, "ALGORITHM_CHOICES")
    eval_choices = parse_tuple_string_assignment(eval_text, "ALGORITHM_CHOICES")
    controlled_supports_sb3 = any(
        algorithm.lower() in {choice.lower() for choice in train_choices + eval_choices}
        for algorithm in ("ppo", "a2c", "ddpg", "sac", "td3", "dqn", "tqc", "trpo", "ars", "qr-dqn")
    ) or "stable_baselines3" in train_text or "stable_baselines3" in eval_text

    checkpoint_notes = []
    if "policy.save(str(save_path / \"model.best\"))" in train_text or "model.best" in train_text:
        checkpoint_notes.append("Training saves best checkpoint with model.best prefix.")
    if all(suffix in eval_text for suffix in ("_actor", "_critic", "_actor_optimizer", "_critic_optimizer")):
        checkpoint_notes.append("Evaluation normalises actor/critic/optimizer suffixes back to checkpoint prefix.")
    if "policy.load(str(checkpoint_prefix))" in eval_text:
        checkpoint_notes.append("Evaluation loads the resolved checkpoint prefix through policy.load(...).")

    scalar_stats_dynamic = "def scalar_stats" in eval_text and "np.isscalar" in eval_text and "row.update(scalar_stats" in eval_text

    rows: List[Dict[str, object]] = [
        {
            "category": "controlled_training_entrypoint",
            "source_path": relative_path(train_path, repo_root),
            "algorithm_group": "controlled TD3-GNN",
            "algorithm": "; ".join(train_choices),
            "appears_implemented": "yes" if train_choices else "no",
            "supported_by_controlled_train": "; ".join(train_choices),
            "supported_by_controlled_eval": "",
            "actiongnn_supported": "yes" if "actiongnn" in train_choices else "no",
            "hierarchical_supported": "yes" if "hierarchical" in train_choices else "no",
            "sb3_supported_by_controlled_pipeline": "yes" if controlled_supports_sb3 else "no",
            "likely_continuous_action_compatible": "yes",
            "requires_discrete_action_wrapper": "no",
            "uses_action_masker_or_special_wrapper": "no",
            "compatible_with_controlled_eval30_csv": "yes for controlled TD3-GNN algorithms only",
            "checkpoint_expectation": "",
            "adapter_required_before_fair_comparison": "",
            "notes": "Controlled training choices are parsed from ALGORITHM_CHOICES.",
        },
        {
            "category": "controlled_evaluation_entrypoint",
            "source_path": relative_path(eval_path, repo_root),
            "algorithm_group": "controlled TD3-GNN",
            "algorithm": "; ".join(eval_choices),
            "appears_implemented": "yes" if eval_choices else "no",
            "supported_by_controlled_train": "",
            "supported_by_controlled_eval": "; ".join(eval_choices),
            "actiongnn_supported": "yes" if "actiongnn" in eval_choices else "no",
            "hierarchical_supported": "yes" if "hierarchical" in eval_choices else "no",
            "sb3_supported_by_controlled_pipeline": "yes" if controlled_supports_sb3 else "no",
            "likely_continuous_action_compatible": "yes",
            "requires_discrete_action_wrapper": "no",
            "uses_action_masker_or_special_wrapper": "no",
            "compatible_with_controlled_eval30_csv": "yes for controlled TD3-GNN algorithms only",
            "checkpoint_expectation": " ".join(checkpoint_notes),
            "adapter_required_before_fair_comparison": "",
            "notes": "Controlled evaluation choices are parsed from ALGORITHM_CHOICES; legacy labels are aliases only.",
        },
        {
            "category": "controlled_evaluation_stats",
            "source_path": relative_path(eval_path, repo_root),
            "algorithm_group": "controlled TD3-GNN",
            "algorithm": "; ".join(eval_choices),
            "appears_implemented": "yes",
            "supported_by_controlled_train": "",
            "supported_by_controlled_eval": "; ".join(eval_choices),
            "actiongnn_supported": "yes" if "actiongnn" in eval_choices else "no",
            "hierarchical_supported": "yes" if "hierarchical" in eval_choices else "no",
            "sb3_supported_by_controlled_pipeline": "yes" if controlled_supports_sb3 else "no",
            "likely_continuous_action_compatible": "yes",
            "requires_discrete_action_wrapper": "no",
            "uses_action_masker_or_special_wrapper": "no",
            "compatible_with_controlled_eval30_csv": "yes",
            "checkpoint_expectation": "",
            "adapter_required_before_fair_comparison": "",
            "notes": "Records scalar EV2Gym env.step stats dynamically." if scalar_stats_dynamic else "Scalar EV2Gym stats are not detected as dynamic CSV columns.",
        },
    ]

    for spec in SB3_ALGORITHM_SPECS:
        appears_implemented = branch_is_implemented(sb3_text, str(spec["branch"])) and class_is_imported(sb3_text, str(spec["class_name"]))
        rows.append({
            "category": "sb3_baseline_candidate",
            "source_path": relative_path(sb3_path, repo_root),
            "algorithm_group": "SB3 MLP candidates",
            "algorithm": spec["display"],
            "appears_implemented": "yes" if appears_implemented else "no",
            "supported_by_controlled_train": "no",
            "supported_by_controlled_eval": "no",
            "actiongnn_supported": "",
            "hierarchical_supported": "",
            "sb3_supported_by_controlled_pipeline": "no",
            "likely_continuous_action_compatible": spec["continuous"],
            "requires_discrete_action_wrapper": spec["requires_discrete_wrapper"],
            "uses_action_masker_or_special_wrapper": spec["special_wrapper"],
            "compatible_with_controlled_eval30_csv": "no",
            "checkpoint_expectation": "train_baselines.py uses EvalCallback best_model_save_path; no controlled eval30 loader is wired here.",
            "adapter_required_before_fair_comparison": spec["adapter_note"],
            "notes": "Repository-supported candidate baseline only; not audited as paper-level SOTA.",
        })

    summary = {
        "train_choices": train_choices,
        "eval_choices": eval_choices,
        "controlled_supports_sb3": controlled_supports_sb3,
        "scalar_stats_dynamic": scalar_stats_dynamic,
        "checkpoint_notes": checkpoint_notes,
        "sb3_detected": [row["algorithm"] for row in rows if row["category"] == "sb3_baseline_candidate" and row["appears_implemented"] == "yes"],
    }
    return rows, summary


def audit_metric_inventory(repo_root: Path) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    eval_path = repo_root / "evaluate_td3_gnn.py"
    aggregate_path = repo_root / "scripts" / "aggregate_eval30_multimetric.py"
    eval_text = read_text(eval_path)
    aggregate_text = read_text(aggregate_path)

    aggregate_key_metrics = parse_list_or_set_strings(aggregate_text, "KEY_METRICS")
    aggregate_required_columns = set(parse_list_or_set_strings(aggregate_text, "REQUIRED_COLUMNS"))
    lower_is_better = set(parse_list_or_set_strings(aggregate_text, "LOWER_IS_BETTER"))
    higher_is_better = set(parse_list_or_set_strings(aggregate_text, "HIGHER_IS_BETTER"))
    scalar_stats_dynamic = "def scalar_stats" in eval_text and "np.isscalar" in eval_text and "row.update(scalar_stats" in eval_text
    action_diagnostics_present = all(metric in eval_text for metric in ACTION_DIAGNOSTIC_METRICS)

    rows: List[Dict[str, object]] = []
    for metric in KEY_METRICS_TO_AUDIT:
        if metric == "episode_reward":
            eval_records = "explicit"
        elif metric in ACTION_DIAGNOSTIC_METRICS and action_diagnostics_present:
            eval_records = "explicit_action_diagnostic"
        elif scalar_stats_dynamic:
            eval_records = "dynamic_scalar_env_step_stat"
        else:
            eval_records = "not_detected"

        if metric in lower_is_better:
            direction = "lower_is_better"
        elif metric in higher_is_better:
            direction = "higher_is_better"
        else:
            direction = "context_dependent_or_unlabelled"

        rows.append({
            "metric_name": metric,
            "kind": "key_metric",
            "evaluate_td3_gnn_records": eval_records,
            "aggregate_eval30_multimetric_handles": "yes" if metric in aggregate_key_metrics else "no",
            "aggregate_required_column": "yes" if metric in aggregate_required_columns or metric in aggregate_key_metrics else "no",
            "preferred_direction": direction,
            "source_path": f"{relative_path(eval_path, repo_root)}; {relative_path(aggregate_path, repo_root)}",
            "notes": "Aggregator metric support is parsed from KEY_METRICS.",
        })

    for metric in ("per_transformer_action_saturation", "per_charger_action_saturation"):
        rows.append({
            "metric_name": metric,
            "kind": "missing_diagnostic",
            "evaluate_td3_gnn_records": "no",
            "aggregate_eval30_multimetric_handles": "no",
            "aggregate_required_column": "no",
            "preferred_direction": "lower_is_better",
            "source_path": relative_path(eval_path, repo_root),
            "notes": "Current evaluator records aggregate action saturation only; per-transformer/per-charger columns are not detected.",
        })

    rows.append({
        "metric_name": "overload_count_vs_magnitude_distinction",
        "kind": "missing_diagnostic",
        "evaluate_td3_gnn_records": "no",
        "aggregate_eval30_multimetric_handles": "no",
        "aggregate_required_column": "no",
        "preferred_direction": "context_dependent_or_unlabelled",
        "source_path": f"{relative_path(eval_path, repo_root)}; {relative_path(aggregate_path, repo_root)}",
        "notes": "Current columns expose total_transformer_overload, but no separate per-transformer overload count and overload magnitude columns are detected.",
    })

    summary = {
        "aggregate_key_metrics": aggregate_key_metrics,
        "scalar_stats_dynamic": scalar_stats_dynamic,
        "action_diagnostics_present": action_diagnostics_present,
        "missing_diagnostics": [
            "per-transformer action saturation",
            "per-charger action saturation",
            "overload count vs overload magnitude distinction",
        ],
    }
    return rows, summary


def detect_scales(text: str, path: Path) -> List[str]:
    combined_text = f"{path.as_posix()} {text}".lower()
    scales = []
    for scale in EXPECTED_SCALES:
        scale_number = scale.replace("CP", "").lower()
        if re.search(rf"\b{scale_number}\s*cp\b", combined_text) or re.search(rf"\b{scale_number}cp\b", combined_text):
            scales.append(scale)
    return scales


def detect_algorithms(text: str) -> List[str]:
    lower_text = text.lower()
    algorithms = []
    if "actiongnn" in lower_text or "baseline" in lower_text:
        algorithms.append("actiongnn")
    if "hierarchical" in lower_text:
        algorithms.append("hierarchical")
    if "sb3" in lower_text or "stable-baselines" in lower_text or "stable_baselines" in lower_text:
        algorithms.append("SB3")
    return algorithms


def classify_protocol(text: str, path: Path, scales: Sequence[str]) -> str:
    lower_text = f"{path.name} {text}".lower()
    filename = path.name.lower()
    has_explicit_1000cp = "1000CP" in scales

    if "formal_eval30" in filename or ("eval30" in filename and "formal" in filename):
        return "formal_eval30"
    if "formal_train" in filename or ("formal" in filename and "train" in filename):
        return "formal_training"
    if "runtime_pilot" in filename or ("runtime" in filename and "pilot" in filename):
        return "runtime_pilot"
    if "profile" in filename or "profiling" in filename:
        return "runtime_profile"
    if "smoke" in filename:
        return "smoke"

    if has_explicit_1000cp and any(term in lower_text for term in ("not yet formal", "no feasibility", "stretch", "smoke")):
        return "1000cp_gap_or_feasibility_note"
    if "formal" in lower_text and "eval30" in lower_text:
        return "formal_eval30"
    if "formal" in lower_text and "train" in lower_text:
        return "formal_training"
    if "aggregation" in lower_text or "aggregate" in lower_text:
        return "aggregation"
    if "runtime" in lower_text and "pilot" in lower_text:
        return "runtime_pilot"
    if "profile" in lower_text or "profiling" in lower_text:
        return "runtime_profile"
    if "smoke" in lower_text:
        return "smoke"
    if "formal evidence" in lower_text:
        return "evidence_note"
    return "related_protocol_or_note"


def audit_m3_inventory(repo_root: Path) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    candidate_paths = []
    m3_dir = repo_root / "m3_jobs"
    docs_dir = repo_root / "docs"
    if m3_dir.exists():
        candidate_paths.extend(sorted(m3_dir.glob("*.slurm")))
    if docs_dir.exists():
        candidate_paths.extend(sorted(docs_dir.glob("*.md")))

    rows: List[Dict[str, object]] = []
    for candidate_path in candidate_paths:
        text = read_text(candidate_path)
        lower_text = text.lower()
        scales = detect_scales(text, candidate_path)
        algorithms = detect_algorithms(text)
        mentions_relevant_protocol = (
            scales
            or any(term in lower_text for term in ("formal", "eval30", "runtime", "profile", "smoke", "sb3", "1000cp"))
        )
        if not mentions_relevant_protocol:
            continue

        protocol_category = classify_protocol(text, candidate_path, scales)
        rows.append({
            "category": protocol_category,
            "path": relative_path(candidate_path, repo_root),
            "kind": "m3_job" if candidate_path.suffix == ".slurm" else "doc",
            "scales": "; ".join(scales),
            "algorithms": "; ".join(algorithms),
            "mentions_train": "yes" if "train" in lower_text or "training" in lower_text else "no",
            "mentions_eval30": "yes" if "eval30" in lower_text else "no",
            "mentions_formal": "yes" if "formal" in lower_text else "no",
            "mentions_runtime": "yes" if "runtime" in lower_text or "wall-time" in lower_text or "wall time" in lower_text else "no",
            "mentions_smoke": "yes" if "smoke" in lower_text else "no",
            "notes": "Static inventory only; this audit does not modify or submit jobs.",
        })

    formal_train_eval_by_scale: Dict[str, bool] = {scale: False for scale in EXPECTED_SCALES}
    smoke_or_runtime_by_scale: Dict[str, bool] = {scale: False for scale in EXPECTED_SCALES}
    for row in rows:
        row_scales = {scale.strip() for scale in str(row["scales"]).split(";") if scale.strip()}
        for scale in row_scales:
            if row["kind"] == "m3_job" and row["mentions_formal"] == "yes" and (
                row["mentions_train"] == "yes" or row["mentions_eval30"] == "yes"
            ):
                formal_train_eval_by_scale[scale] = True
            if row["kind"] == "m3_job" and (
                row["mentions_smoke"] == "yes" or row["mentions_runtime"] == "yes" or row["category"] == "runtime_profile"
            ):
                smoke_or_runtime_by_scale[scale] = True

    summary = {
        "formal_train_eval_by_scale": formal_train_eval_by_scale,
        "smoke_or_runtime_by_scale": smoke_or_runtime_by_scale,
        "m3_job_count": sum(1 for row in rows if row["kind"] == "m3_job"),
        "doc_count": sum(1 for row in rows if row["kind"] == "doc"),
    }
    return rows, summary


def build_readiness_matrix(
    config_summary: Dict[str, object],
    algorithm_summary: Dict[str, object],
    m3_summary: Dict[str, object],
) -> List[Dict[str, object]]:
    present_scales = set(config_summary["present_scales"])
    train_choices = set(algorithm_summary["train_choices"])
    eval_choices = set(algorithm_summary["eval_choices"])
    formal_m3 = m3_summary["formal_train_eval_by_scale"]
    smoke_or_runtime_m3 = m3_summary["smoke_or_runtime_by_scale"]

    rows: List[Dict[str, object]] = []
    for scale in EXPECTED_SCALES:
        config_present = scale in present_scales
        for algorithm_group, algorithm in (
            ("controlled ActionGNN", "actiongnn"),
            ("controlled hierarchical", "hierarchical"),
        ):
            controlled_train_supported = algorithm in train_choices
            controlled_eval_supported = algorithm in eval_choices
            if not config_present:
                classification = "missing_config"
            elif not controlled_train_supported or not controlled_eval_supported:
                classification = "not_currently_supported"
            elif scale == "1000CP" and not smoke_or_runtime_m3.get(scale, False):
                classification = "needs_smoke_test"
            else:
                classification = "ready_for_formal_train_eval"

            notes = []
            if scale == "25CP":
                notes.append("Formal evidence is documented; no active 25CP M3 formal job script was detected in m3_jobs.")
            if scale == "100CP":
                notes.append("Formal/runtime protocols exist, but paired-seed reward robustness should be treated carefully.")
            if scale == "500CP":
                notes.append("Formal train/eval30 protocols and multimetric aggregation support are detected.")
            if scale == "1000CP":
                notes.append("Config exists only if detected above; smoke, runtime, and feasibility validation must precede formal jobs.")

            rows.append({
                "scale": scale,
                "algorithm_group": algorithm_group,
                "classification": classification,
                "config_present": "yes" if config_present else "no",
                "controlled_train_supported": "yes" if controlled_train_supported else "no",
                "controlled_eval_supported": "yes" if controlled_eval_supported else "no",
                "sb3_adapter_present": "not_applicable",
                "m3_protocol_support": "formal_train_eval" if formal_m3.get(scale, False) else "smoke_or_runtime" if smoke_or_runtime_m3.get(scale, False) else "none_detected",
                "notes": " ".join(notes),
            })

        sb3_classification = "missing_config" if not config_present else "needs_adapter"
        sb3_notes = "SB3 candidates are implemented in train_baselines.py, but not integrated into controlled deterministic eval30 CSVs."
        if scale == "1000CP":
            sb3_notes += " 1000CP also requires smoke, memory, runtime, and wrapper feasibility validation."
        rows.append({
            "scale": scale,
            "algorithm_group": "SB3 MLP candidates",
            "classification": sb3_classification,
            "config_present": "yes" if config_present else "no",
            "controlled_train_supported": "no",
            "controlled_eval_supported": "no",
            "sb3_adapter_present": "no",
            "m3_protocol_support": "none_detected_for_sb3",
            "notes": sb3_notes,
        })

    return rows


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> List[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for header in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def write_summary(
    path: Path,
    config_summary: Dict[str, object],
    algorithm_summary: Dict[str, object],
    metric_summary: Dict[str, object],
    m3_summary: Dict[str, object],
    readiness_rows: Sequence[Dict[str, object]],
) -> None:
    present_scales = config_summary["present_scales"]
    missing_scales = config_summary["missing_expected_scales"]
    canonical_paths_by_scale = config_summary["canonical_paths_by_scale"]
    sb3_detected = algorithm_summary["sb3_detected"]

    lines: List[str] = [
        "# Multi-scale Multi-algorithm Experiment Matrix Audit",
        "",
        "This is a static repository audit. It does not launch jobs, train policies, evaluate checkpoints, or modify experiment behaviour.",
        "",
        "## What is currently ready",
        "",
        "- The controlled TD3-GNN training entry point supports: "
        + ", ".join(algorithm_summary["train_choices"])
        + ".",
        "- The controlled TD3-GNN evaluation entry point supports: "
        + ", ".join(algorithm_summary["eval_choices"])
        + ".",
        "- PublicPST scale configs detected: " + (", ".join(present_scales) if present_scales else "none") + ".",
        "- `evaluate_td3_gnn.py` records reward and aggregate action diagnostics, and "
        + ("dynamically records scalar EV2Gym step stats." if algorithm_summary["scalar_stats_dynamic"] else "does not appear to dynamically record scalar EV2Gym step stats."),
        "- `scripts/aggregate_eval30_multimetric.py` handles the current 500CP multimetric ActionGNN-vs-hierarchical CSV protocol.",
        "",
        "## What is not ready",
        "",
        "- SB3 baselines are not integrated into the manifest-controlled deterministic eval30 CSV pipeline.",
        "- 1000CP should not be treated as formal-ready until smoke, runtime, memory, and feasibility checks pass.",
        "- Per-transformer and per-charger action saturation diagnostics are not currently emitted by the controlled evaluator.",
        "- Current CSV columns do not distinguish transformer overload count from overload magnitude.",
        "",
        "## Config inventory summary",
        "",
    ]

    lines.extend(markdown_table(
        ("scale", "detected paths"),
        [
            (
                scale,
                ", ".join(canonical_paths_by_scale[scale]) if canonical_paths_by_scale[scale] else "missing",
            )
            for scale in EXPECTED_SCALES
        ],
    ))
    lines.extend([
        "",
        "Missing expected scales: " + (", ".join(missing_scales) if missing_scales else "none") + ".",
        "",
        "## Controlled pipeline boundary",
        "",
        "- Controlled training algorithms: " + ", ".join(algorithm_summary["train_choices"]) + ".",
        "- Controlled evaluation algorithms: " + ", ".join(algorithm_summary["eval_choices"]) + ".",
        "- SB3 support in controlled eval30: no.",
        "- Checkpoint expectation: " + (" ".join(algorithm_summary["checkpoint_notes"]) if algorithm_summary["checkpoint_notes"] else "not detected"),
        "",
        "## SB3 candidate baseline inventory",
        "",
        "Detected repository-supported candidate baselines: " + (", ".join(sb3_detected) if sb3_detected else "none") + ".",
        "",
        "Before fair external-baseline comparison, the repository needs:",
        "",
        "- SB3 checkpoint save/load adapter.",
        "- Deterministic eval30 adapter for SB3 policies.",
        "- Unified canonical CSV schema.",
        "- Common episode seed protocol.",
        "- Wrapper and action-space metadata in the manifest.",
        "- Source manifest and artifact packaging for SB3 runs.",
        "",
        "These candidates are not labelled as paper-level SOTA baselines by this audit.",
        "",
        "## Metric support",
        "",
        "- Reward alone is insufficient for paper-level claims.",
        "- Operational metrics handled by the current aggregation include: "
        + ", ".join(metric for metric in KEY_METRICS_TO_AUDIT if metric in metric_summary["aggregate_key_metrics"])
        + ".",
        "- Missing diagnostics: " + ", ".join(metric_summary["missing_diagnostics"]) + ".",
        "",
        "## M3 and protocol inventory",
        "",
        "- M3 job files inventoried: " + str(m3_summary["m3_job_count"]) + ".",
        "- Documentation files inventoried: " + str(m3_summary["doc_count"]) + ".",
        "- Formal train/eval support by scale: "
        + ", ".join(
            f"{scale}={'yes' if m3_summary['formal_train_eval_by_scale'].get(scale, False) else 'no'}"
            for scale in EXPECTED_SCALES
        )
        + ".",
        "- Smoke/runtime/profile support by scale: "
        + ", ".join(
            f"{scale}={'yes' if m3_summary['smoke_or_runtime_by_scale'].get(scale, False) else 'no'}"
            for scale in EXPECTED_SCALES
        )
        + ".",
        "",
        "## Readiness matrix",
        "",
    ])

    lines.extend(markdown_table(
        ("scale", "algorithm group", "classification"),
        [(row["scale"], row["algorithm_group"], row["classification"]) for row in readiness_rows],
    ))

    lines.extend([
        "",
        "## Recommended next stages",
        "",
        "### Stage 0: repository audit",
        "",
        "- Run this audit script.",
        "- Verify configs, algorithms, metrics, M3 scripts, and protocol documents.",
        "",
        "### Stage 1: smoke matrix",
        "",
        "- Use one seed.",
        "- Use short training or checkpoint-load tests.",
        "- Use 1-3 evaluation episodes.",
        "- Detect crash, memory, action-space, wrapper, checkpoint, and CSV-schema issues.",
        "",
        "### Stage 2: pilot matrix",
        "",
        "- Use selected scales and selected algorithms.",
        "- Use 1-2 seeds.",
        "- Run eval30.",
        "- Verify metric schema and aggregation across scale and algorithm labels.",
        "",
        "### Stage 3: formal matrix",
        "",
        "- Use 5 matched seeds.",
        "- Run eval30.",
        "- Package complete evidence artifacts.",
        "- Run statistical aggregation only after all canonical CSVs are fixed.",
        "",
        "## Explicit warning",
        "",
        "Do not launch a full 25/100/500/1000CP x multi-algorithm x five-seed job before smoke and adapter validation pass.",
        "",
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config_rows, config_summary = audit_config_inventory(repo_root)
    algorithm_rows, algorithm_summary = audit_algorithm_inventory(repo_root)
    metric_rows, metric_summary = audit_metric_inventory(repo_root)
    m3_rows, m3_summary = audit_m3_inventory(repo_root)
    readiness_rows = build_readiness_matrix(config_summary, algorithm_summary, m3_summary)

    write_csv(
        out_dir / "experiment_matrix_config_inventory.csv",
        config_rows,
        [
            "record_type",
            "path",
            "exists",
            "inferred_scale",
            "scale_from_charging_station_count",
            *CONFIG_FIELDS,
            "appears_usable_for_formal_controlled_experiments",
            "missing_required_fields",
            "missing_expected_scales",
            "notes",
        ],
    )
    write_csv(
        out_dir / "experiment_matrix_algorithm_inventory.csv",
        algorithm_rows,
        [
            "category",
            "source_path",
            "algorithm_group",
            "algorithm",
            "appears_implemented",
            "supported_by_controlled_train",
            "supported_by_controlled_eval",
            "actiongnn_supported",
            "hierarchical_supported",
            "sb3_supported_by_controlled_pipeline",
            "likely_continuous_action_compatible",
            "requires_discrete_action_wrapper",
            "uses_action_masker_or_special_wrapper",
            "compatible_with_controlled_eval30_csv",
            "checkpoint_expectation",
            "adapter_required_before_fair_comparison",
            "notes",
        ],
    )
    write_csv(
        out_dir / "experiment_matrix_metric_inventory.csv",
        metric_rows,
        [
            "metric_name",
            "kind",
            "evaluate_td3_gnn_records",
            "aggregate_eval30_multimetric_handles",
            "aggregate_required_column",
            "preferred_direction",
            "source_path",
            "notes",
        ],
    )
    write_csv(
        out_dir / "experiment_matrix_m3_inventory.csv",
        m3_rows,
        [
            "category",
            "path",
            "kind",
            "scales",
            "algorithms",
            "mentions_train",
            "mentions_eval30",
            "mentions_formal",
            "mentions_runtime",
            "mentions_smoke",
            "notes",
        ],
    )
    write_csv(
        out_dir / "experiment_matrix_readiness.csv",
        readiness_rows,
        [
            "scale",
            "algorithm_group",
            "classification",
            "config_present",
            "controlled_train_supported",
            "controlled_eval_supported",
            "sb3_adapter_present",
            "m3_protocol_support",
            "notes",
        ],
    )
    write_summary(
        out_dir / "experiment_matrix_audit_summary.md",
        config_summary,
        algorithm_summary,
        metric_summary,
        m3_summary,
        readiness_rows,
    )

    for output_name in (
        "experiment_matrix_config_inventory.csv",
        "experiment_matrix_algorithm_inventory.csv",
        "experiment_matrix_metric_inventory.csv",
        "experiment_matrix_m3_inventory.csv",
        "experiment_matrix_readiness.csv",
        "experiment_matrix_audit_summary.md",
    ):
        print(out_dir / output_name)


if __name__ == "__main__":
    main()
