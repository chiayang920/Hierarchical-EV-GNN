from pathlib import Path
import importlib
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_controlled_evaluator_imports():
    module = importlib.import_module("evaluator_td3_actiongnn_controlled")
    assert module is not None


def test_policy_factory_supports_baseline_and_hierarchical():
    module = importlib.import_module("evaluator_td3_actiongnn_controlled")

    baseline_policy = module.create_policy(
        algorithm="baseline_25cp",
        action_dim=25,
        max_action=1.0,
        device="cpu",
    )
    hierarchical_policy = module.create_policy(
        algorithm="hierarchical_25cp",
        action_dim=25,
        max_action=1.0,
        device="cpu",
    )

    assert baseline_policy.__class__.__name__ == "TD3_ActionGNN"
    assert hierarchical_policy.__class__.__name__ == "TD3_HierarchicalActionGNN"


def test_policy_factory_rejects_invalid_algorithm():
    module = importlib.import_module("evaluator_td3_actiongnn_controlled")

    with pytest.raises(ValueError, match="Unsupported algorithm"):
        module.create_policy(
            algorithm="not_a_policy",
            action_dim=25,
            max_action=1.0,
            device="cpu",
        )


class RecordingPolicy:
    def __init__(self):
        self.expl_noise_values = []

    def select_action(self, state, expl_noise=0.0, return_mapped_action=False):
        self.expl_noise_values.append(expl_noise)
        assert return_mapped_action is True
        return np.array([0.0, 1.0], dtype=np.float32), "full_node_action"


def test_select_mapped_action_respects_deterministic_and_eval_noise():
    module = importlib.import_module("evaluator_td3_actiongnn_controlled")

    deterministic_policy = RecordingPolicy()
    deterministic_action = module.select_mapped_action(
        policy=deterministic_policy,
        state=object(),
        deterministic=True,
        eval_expl_noise=0.25,
    )

    stochastic_policy = RecordingPolicy()
    stochastic_action = module.select_mapped_action(
        policy=stochastic_policy,
        state=object(),
        deterministic=False,
        eval_expl_noise=0.25,
    )

    assert deterministic_policy.expl_noise_values == [0.0]
    assert stochastic_policy.expl_noise_values == [0.25]
    assert deterministic_action.dtype == np.float32
    assert stochastic_action.dtype == np.float32


def test_action_diagnostics_from_mapped_actions():
    module = importlib.import_module("evaluator_td3_actiongnn_controlled")
    mapped_actions = [
        np.array([0.0, 1.0, 1.0], dtype=np.float32),
        np.array([0.5, 0.0, 1.0], dtype=np.float32),
    ]

    diagnostics = module.action_diagnostics_from_actions(mapped_actions, max_action=1.0)

    assert diagnostics["action_mean"] == pytest.approx(3.5 / 6.0)
    assert diagnostics["action_min"] == 0.0
    assert diagnostics["action_max"] == 1.0
    assert diagnostics["action_fraction_zero"] == pytest.approx(2.0 / 6.0)
    assert diagnostics["action_fraction_at_max"] == pytest.approx(3.0 / 6.0)
    assert diagnostics["active_action_count_mean"] == 2.0


def test_csv_rows_include_required_schema_and_scalar_stats():
    module = importlib.import_module("evaluator_td3_actiongnn_controlled")

    metadata = {
        "run_name": "contract_test",
        "algorithm": "hierarchical_25cp",
        "config": "./config_files/PublicPST_25cp.yaml",
        "seed": 7,
        "checkpoint": "model.best",
    }
    episode_records = [
        {
            "episode_index": 0,
            "episode_seed": 100007,
            "episode_reward": -10.0,
            "episode_steps": 112,
            "done": True,
            "action_mean": 0.1,
            "action_std": 0.2,
            "action_min": 0.0,
            "action_max": 1.0,
            "action_fraction_zero": 0.3,
            "action_fraction_at_max": 0.4,
            "active_action_count_mean": 5.0,
            "stats": {
                "tracking_error": 10.0,
                "non_scalar_metric": {"nested": 1},
            },
        },
        {
            "episode_index": 1,
            "episode_seed": 100008,
            "episode_reward": -14.0,
            "episode_steps": 112,
            "done": True,
            "action_mean": 0.2,
            "action_std": 0.3,
            "action_min": 0.0,
            "action_max": 1.0,
            "action_fraction_zero": 0.4,
            "action_fraction_at_max": 0.5,
            "active_action_count_mean": 6.0,
            "stats": {"tracking_error": 14.0},
        },
    ]

    rows, fieldnames = module.build_csv_rows(metadata, episode_records)
    required_columns = {
        "run_name",
        "algorithm",
        "config",
        "seed",
        "episode_seed",
        "checkpoint",
        "episode_index",
        "episode_reward",
        "episode_steps",
        "done",
        "mean_reward",
        "std_reward",
        "action_mean",
        "action_std",
        "action_min",
        "action_max",
        "action_fraction_zero",
        "action_fraction_at_max",
        "active_action_count_mean",
    }

    assert required_columns.issubset(set(fieldnames))
    assert "tracking_error" in fieldnames
    assert "non_scalar_metric" not in fieldnames
    assert len(rows) == 3
    assert rows[0]["mean_reward"] == -12.0
    assert rows[0]["episode_seed"] == 100007
    assert rows[2]["episode_index"] == "summary"
