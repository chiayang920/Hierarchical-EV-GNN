import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_MODULE_PATH = PROJECT_ROOT / "train_td3_gnn.py"


def load_training_module(monkeypatch):
    training_utils = ModuleType("utils.ev2gym_training_utils")
    training_utils.make_env = lambda *_args, **_kwargs: None
    training_utils.normalise_step_result = lambda result: result
    training_utils.reset_env = lambda *_args, **_kwargs: (None, {})
    training_utils.resolve_device = lambda value: value
    training_utils.set_global_seed = lambda _seed: None
    training_utils.str2bool = lambda value: bool(value)
    monkeypatch.setitem(sys.modules, "utils.ev2gym_training_utils", training_utils)

    module_name = "train_td3_gnn_logging_contract"
    spec = importlib.util.spec_from_file_location(module_name, TRAINING_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_raw_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def read_dict_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_write_log_row_expands_header_for_evaluation_fields(tmp_path, monkeypatch):
    module = load_training_module(monkeypatch)
    log_path = tmp_path / "training_log.csv"

    module.write_log_row(
        log_path,
        {
            "type": "train_episode",
            "timestep": 112,
            "episode": 0,
            "episode_timesteps": 112,
            "episode_reward": -100.0,
            "critic_loss": 1.0,
            "actor_loss": 2.0,
            "elapsed_seconds": 3.0,
        },
    )
    module.write_log_row(
        log_path,
        {
            "type": "evaluation",
            "timestep": 256,
            "episode": 2,
            "episode_timesteps": 32,
            "episode_reward": -50.0,
            "critic_loss": 4.0,
            "actor_loss": 5.0,
            "elapsed_seconds": 6.0,
            "eval/mean_reward": -110560.78134400371,
            "eval/std_reward": 0.0,
            "eval/best_reward": -110560.78134400371,
        },
    )

    raw_rows = read_raw_rows(log_path)
    header = raw_rows[0]
    assert "eval/mean_reward" in header
    assert all(len(row) == len(header) for row in raw_rows)

    dict_rows = read_dict_rows(log_path)
    assert dict_rows[0]["eval/mean_reward"] == ""
    assert dict_rows[1]["eval/mean_reward"] == "-110560.78134400371"
    assert None not in dict_rows[1]


def test_write_log_row_preserves_rows_when_later_metrics_expand_schema(tmp_path, monkeypatch):
    module = load_training_module(monkeypatch)
    log_path = tmp_path / "training_log.csv"

    module.write_log_row(log_path, {"type": "train_episode", "timestep": 1})
    module.write_log_row(
        log_path,
        {
            "type": "evaluation",
            "timestep": 2,
            "eval/mean_reward": -10.0,
        },
    )
    module.write_log_row(
        log_path,
        {
            "type": "final_evaluation",
            "timestep": 2,
            "eval/mean_reward": -9.0,
            "eval_metrics/power_tracker_violation_mean": 3.5,
        },
    )

    raw_rows = read_raw_rows(log_path)
    header = raw_rows[0]
    assert header == [
        "type",
        "timestep",
        "eval/mean_reward",
        "eval_metrics/power_tracker_violation_mean",
    ]
    assert all(len(row) == len(header) for row in raw_rows)

    dict_rows = read_dict_rows(log_path)
    assert dict_rows[0]["eval/mean_reward"] == ""
    assert dict_rows[1]["eval_metrics/power_tracker_violation_mean"] == ""
    assert dict_rows[2]["eval_metrics/power_tracker_violation_mean"] == "3.5"
    assert all(None not in row for row in dict_rows)


def test_write_log_row_rejects_preexisting_unnamed_extra_columns(tmp_path, monkeypatch):
    module = load_training_module(monkeypatch)
    log_path = tmp_path / "training_log.csv"
    log_path.write_text(
        "type,timestep\n"
        "evaluation,256,-110560.78134400371\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unnamed extra columns"):
        module.write_log_row(
            log_path,
            {
                "type": "final_evaluation",
                "timestep": 512,
                "eval/mean_reward": -110560.78134400371,
            },
        )
