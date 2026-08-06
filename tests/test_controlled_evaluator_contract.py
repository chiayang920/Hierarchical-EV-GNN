from pathlib import Path
import importlib
import sys

import numpy as np
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_controlled_evaluator_imports():
    module = importlib.import_module("evaluate_td3_gnn")
    assert module is not None


def test_policy_factory_supports_actiongnn_and_hierarchical():
    module = importlib.import_module("evaluate_td3_gnn")

    actiongnn_policy = module.create_policy(
        algorithm="actiongnn",
        action_dim=25,
        max_action=1.0,
        device="cpu",
    )
    hierarchical_policy = module.create_policy(
        algorithm="hierarchical",
        action_dim=25,
        max_action=1.0,
        device="cpu",
    )

    assert actiongnn_policy.__class__.__name__ == "TD3_ActionGNN"
    assert hierarchical_policy.__class__.__name__ == "TD3_HierarchicalActionGNN"


def test_policy_factory_supports_actiongnn_nonnegative():
    module = importlib.import_module("evaluate_td3_gnn")

    corrected_policy = module.create_policy(
        algorithm="actiongnn_nonnegative",
        action_dim=25,
        max_action=1.0,
        device="cpu",
    )

    assert corrected_policy.__class__.__name__ == "TD3_ActionGNN_NonNegative"


def test_legacy_algorithm_aliases_normalise_to_canonical_labels():
    module = importlib.import_module("evaluate_td3_gnn")

    with pytest.warns(DeprecationWarning, match="baseline_25cp"):
        assert module.normalise_algorithm_label("baseline_25cp") == "actiongnn"
    with pytest.warns(DeprecationWarning, match="hierarchical_25cp"):
        assert module.normalise_algorithm_label("hierarchical_25cp") == "hierarchical"


def test_actiongnn_nonnegative_is_not_a_legacy_alias():
    module = importlib.import_module("evaluate_td3_gnn")

    assert module.normalise_algorithm_label("actiongnn_nonnegative") == "actiongnn_nonnegative"
    assert module.LEGACY_ALGORITHM_ALIASES.get("actiongnn") is None
    assert module.LEGACY_ALGORITHM_ALIASES.get("actiongnn_nonnegative") is None


def test_policy_factory_rejects_invalid_algorithm():
    module = importlib.import_module("evaluate_td3_gnn")

    with pytest.raises(ValueError, match="Unsupported algorithm"):
        module.create_policy(
            algorithm="not_a_policy",
            action_dim=25,
            max_action=1.0,
            device="cpu",
        )


class RecordingLoadPolicy:
    def __init__(self):
        self.loaded_prefixes = []

    def load(self, checkpoint_prefix):
        self.loaded_prefixes.append(checkpoint_prefix)


def write_yaml(path, payload):
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def corrected_metadata(checkpoint_role="best", algorithm="actiongnn_nonnegative"):
    return {
        "metadata_schema": "actiongnn_nonnegative_checkpoint_v1",
        "algorithm": algorithm,
        "action_domain_contract": "nonnegative_ev_rows_exact_zero_nonev_v1",
        "actor_output_transform": "shifted_tanh_v1",
        "actor_output_transform_formula": "0.5 * max_action * (tanh(z) + 1.0)",
        "non_ev_action": 0.0,
        "discrete_actions": 1,
        "training_budget": 50000,
        "start_timesteps": 1000,
        "eval_frequency": 5000,
        "internal_eval_episodes": 5,
        "checkpoint_selection_rule": (
            "model.best selected by strict improvement of scheduled internal eval mean reward "
            "at 5k-step intervals within the 50k training budget; model.last is saved at "
            "50k but is not used for canonical eval30 or diagnostics."
        ),
        "checkpoint_role": checkpoint_role,
    }


def test_load_policy_checkpoint_allows_legacy_checkpoint_without_corrected_metadata(tmp_path):
    module = importlib.import_module("evaluate_td3_gnn")
    checkpoint_prefix = tmp_path / "model.best"
    write_yaml(tmp_path / "run_args.yaml", {"algorithm": "actiongnn"})
    policy = RecordingLoadPolicy()

    module.load_policy_checkpoint(policy, checkpoint_prefix, "actiongnn")

    assert policy.loaded_prefixes == [str(checkpoint_prefix)]


def test_corrected_checkpoint_missing_metadata_is_rejected_before_state_dict_load(tmp_path):
    module = importlib.import_module("evaluate_td3_gnn")
    checkpoint_prefix = tmp_path / "model.best"
    write_yaml(tmp_path / "run_args.yaml", {"algorithm": "actiongnn_nonnegative"})
    policy = RecordingLoadPolicy()

    with pytest.raises(ValueError, match="metadata"):
        module.load_policy_checkpoint(policy, checkpoint_prefix, "actiongnn_nonnegative")

    assert policy.loaded_prefixes == []


def test_corrected_checkpoint_metadata_mismatch_is_rejected_before_state_dict_load(tmp_path):
    module = importlib.import_module("evaluate_td3_gnn")
    checkpoint_prefix = tmp_path / "model.best"
    write_yaml(tmp_path / "run_args.yaml", {"algorithm": "actiongnn_nonnegative"})
    write_yaml(
        tmp_path / "model.best.metadata.yaml",
        corrected_metadata(algorithm="hierarchical"),
    )
    policy = RecordingLoadPolicy()

    with pytest.raises(ValueError, match="metadata.*algorithm"):
        module.load_policy_checkpoint(policy, checkpoint_prefix, "actiongnn_nonnegative")

    assert policy.loaded_prefixes == []


def test_legacy_run_args_algorithm_mismatch_is_rejected_before_state_dict_load(tmp_path):
    module = importlib.import_module("evaluate_td3_gnn")
    checkpoint_prefix = tmp_path / "model.best"
    write_yaml(tmp_path / "run_args.yaml", {"algorithm": "hierarchical"})
    policy = RecordingLoadPolicy()

    with pytest.raises(ValueError, match="run_args.yaml.*algorithm"):
        module.load_policy_checkpoint(policy, checkpoint_prefix, "actiongnn")

    assert policy.loaded_prefixes == []


def test_checkpoint_role_mismatch_is_rejected_before_state_dict_load(tmp_path):
    module = importlib.import_module("evaluate_td3_gnn")
    checkpoint_prefix = tmp_path / "model.best"
    write_yaml(tmp_path / "run_args.yaml", {"algorithm": "actiongnn_nonnegative"})
    write_yaml(tmp_path / "model.best.metadata.yaml", corrected_metadata(checkpoint_role="last"))
    policy = RecordingLoadPolicy()

    with pytest.raises(ValueError, match="checkpoint_role"):
        module.load_policy_checkpoint(policy, checkpoint_prefix, "actiongnn_nonnegative")

    assert policy.loaded_prefixes == []


def test_hierarchical_evaluator_route_remains_supported():
    module = importlib.import_module("evaluate_td3_gnn")

    policy = module.create_policy(
        algorithm="hierarchical",
        action_dim=25,
        max_action=1.0,
        device="cpu",
    )

    assert policy.__class__.__name__ == "TD3_HierarchicalActionGNN"


class RecordingPolicy:
    def __init__(self):
        self.expl_noise_values = []

    def select_action(self, state, expl_noise=0.0, return_mapped_action=False):
        self.expl_noise_values.append(expl_noise)
        assert return_mapped_action is True
        return np.array([0.0, 1.0], dtype=np.float32), "full_node_action"


def test_select_mapped_action_respects_deterministic_and_eval_noise():
    module = importlib.import_module("evaluate_td3_gnn")

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
    module = importlib.import_module("evaluate_td3_gnn")
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
    module = importlib.import_module("evaluate_td3_gnn")

    metadata = {
        "run_name": "contract_test",
        "algorithm": "hierarchical",
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
