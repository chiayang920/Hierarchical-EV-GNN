from argparse import Namespace
import importlib

import pytest
import yaml


def write_yaml(path, payload):
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


class RecordingLoadPolicy:
    def __init__(self):
        self.loaded_prefixes = []

    def load(self, checkpoint_prefix):
        self.loaded_prefixes.append(checkpoint_prefix)


def transformer_ev_metadata(checkpoint_role="best", algorithm="hierarchical_transformer_ev"):
    return {
        "metadata_schema": "hierarchical_transformer_ev_checkpoint_v1",
        "algorithm": algorithm,
        "action_domain_contract": "nonnegative_ev_rows_exact_zero_nonev_transformer_ev_v1",
        "actor_output_transform": "transformer_ev_hierarchy_v1",
        "actor_output_transform_formula": (
            "clip(B_G * w_T(t_i) * w_EV(i | t_i) * g_i, 0, max_action)"
        ),
        "non_ev_action": 0.0,
        "discrete_actions": 1,
        "training_budget": 512,
        "start_timesteps": 64,
        "eval_frequency": 256,
        "internal_eval_episodes": 1,
        "checkpoint_selection_rule": (
            "model.best selected by strict improvement of scheduled internal eval mean reward "
            "at intervals recorded in eval_frequency within the configured training budget; "
            "model.last is saved at the configured final step but is not used for canonical "
            "eval30 or diagnostics."
        ),
        "checkpoint_role": checkpoint_role,
    }


def test_train_policy_factory_accepts_hierarchical_transformer_ev():
    train_module = importlib.import_module("train_td3_gnn")

    policy_class = train_module.get_policy_class("hierarchical_transformer_ev")

    assert policy_class.__name__ == "TD3_TransformerEVActionGNN"


def test_evaluator_policy_factory_accepts_hierarchical_transformer_ev():
    evaluator_module = importlib.import_module("evaluate_td3_gnn")

    policy = evaluator_module.create_policy(
        algorithm="hierarchical_transformer_ev",
        action_dim=25,
        max_action=1.0,
        device="cpu",
    )

    assert policy.__class__.__name__ == "TD3_TransformerEVActionGNN"


def test_diagnostic_evaluator_accepts_hierarchical_transformer_ev_label():
    diagnostic_module = importlib.import_module("evaluate_td3_gnn_infrastructure_diagnostics")

    args = diagnostic_module.parse_args(
        [
            "--algorithm",
            "hierarchical_transformer_ev",
            "--scale",
            "25cp",
            "--config",
            "config_files/PublicPST_25cp.yaml",
            "--checkpoint",
            "model.best",
            "--output_dir",
            "diagnostics",
        ]
    )

    assert args.algorithm == "hierarchical_transformer_ev"


def test_transformer_ev_checkpoint_metadata_records_exact_algorithm():
    metadata_module = importlib.import_module("TD3.TD3_TransformerEVActionGNN")
    args = Namespace(
        discrete_actions=1,
        max_timesteps=512,
        start_timesteps=64,
        eval_freq=256,
        eval_episodes=1,
    )

    metadata = metadata_module.build_checkpoint_metadata(args, "best")

    assert metadata["metadata_schema"] == "hierarchical_transformer_ev_checkpoint_v1"
    assert metadata["algorithm"] == "hierarchical_transformer_ev"
    assert metadata["actor_output_transform_formula"] == (
        "clip(B_G * w_T(t_i) * w_EV(i | t_i) * g_i, 0, max_action)"
    )
    assert "5k-step intervals" not in metadata["checkpoint_selection_rule"]
    assert "eval_frequency" in metadata["checkpoint_selection_rule"]
    assert metadata["checkpoint_role"] == "best"


def test_transformer_ev_checkpoint_missing_metadata_is_rejected_before_state_dict_load(tmp_path):
    evaluator_module = importlib.import_module("evaluate_td3_gnn")
    checkpoint_prefix = tmp_path / "model.best"
    write_yaml(tmp_path / "run_args.yaml", {"algorithm": "hierarchical_transformer_ev"})
    policy = RecordingLoadPolicy()

    with pytest.raises(ValueError, match="hierarchical_transformer_ev.*metadata"):
        evaluator_module.load_policy_checkpoint(
            policy,
            checkpoint_prefix,
            "hierarchical_transformer_ev",
        )

    assert policy.loaded_prefixes == []


def test_transformer_ev_checkpoint_metadata_mismatch_is_rejected_before_state_dict_load(tmp_path):
    evaluator_module = importlib.import_module("evaluate_td3_gnn")
    checkpoint_prefix = tmp_path / "model.best"
    write_yaml(tmp_path / "run_args.yaml", {"algorithm": "hierarchical_transformer_ev"})
    write_yaml(
        tmp_path / "model.best.metadata.yaml",
        transformer_ev_metadata(algorithm="hierarchical"),
    )
    policy = RecordingLoadPolicy()

    with pytest.raises(ValueError, match="metadata.*algorithm"):
        evaluator_module.load_policy_checkpoint(
            policy,
            checkpoint_prefix,
            "hierarchical_transformer_ev",
        )

    assert policy.loaded_prefixes == []


def test_checkpoint_identity_rejects_incompatible_declared_architecture(tmp_path):
    evaluator_module = importlib.import_module("evaluate_td3_gnn")
    checkpoint_prefix = tmp_path / "model.best"
    write_yaml(tmp_path / "run_args.yaml", {"algorithm": "hierarchical"})
    write_yaml(tmp_path / "model.best.metadata.yaml", transformer_ev_metadata())

    with pytest.raises(ValueError, match="run_args.yaml.*algorithm"):
        evaluator_module.validate_checkpoint_identity(
            checkpoint_prefix,
            "hierarchical_transformer_ev",
        )


@pytest.mark.parametrize(
    ("run_args_key", "metadata_key", "run_args_value", "metadata_value"),
    [
        ("max_timesteps", "training_budget", 512, 256),
        ("start_timesteps", "start_timesteps", 64, 32),
        ("eval_freq", "eval_frequency", 256, 128),
        ("eval_episodes", "internal_eval_episodes", 1, 2),
    ],
)
def test_transformer_ev_checkpoint_metadata_must_match_run_args(
    tmp_path,
    run_args_key,
    metadata_key,
    run_args_value,
    metadata_value,
):
    evaluator_module = importlib.import_module("evaluate_td3_gnn")
    checkpoint_prefix = tmp_path / "model.best"
    run_args = {
        "algorithm": "hierarchical_transformer_ev",
        "max_timesteps": 512,
        "start_timesteps": 64,
        "eval_freq": 256,
        "eval_episodes": 1,
    }
    metadata = transformer_ev_metadata()
    run_args[run_args_key] = run_args_value
    metadata[metadata_key] = metadata_value
    write_yaml(tmp_path / "run_args.yaml", run_args)
    write_yaml(tmp_path / "model.best.metadata.yaml", metadata)
    policy = RecordingLoadPolicy()

    with pytest.raises(ValueError, match=f"metadata {metadata_key} mismatch"):
        evaluator_module.load_policy_checkpoint(
            policy,
            checkpoint_prefix,
            "hierarchical_transformer_ev",
        )

    assert policy.loaded_prefixes == []


def test_existing_algorithm_routes_remain_unchanged():
    train_module = importlib.import_module("train_td3_gnn")
    evaluator_module = importlib.import_module("evaluate_td3_gnn")

    assert train_module.get_policy_class("actiongnn").__name__ == "TD3_ActionGNN"
    assert (
        train_module.get_policy_class("actiongnn_nonnegative").__name__
        == "TD3_ActionGNN_NonNegative"
    )
    assert (
        train_module.get_policy_class("hierarchical").__name__
        == "TD3_HierarchicalActionGNN"
    )
    assert evaluator_module.get_policy_class("actiongnn").__name__ == "TD3_ActionGNN"
    assert (
        evaluator_module.get_policy_class("actiongnn_nonnegative").__name__
        == "TD3_ActionGNN_NonNegative"
    )
    assert (
        evaluator_module.get_policy_class("hierarchical").__name__
        == "TD3_HierarchicalActionGNN"
    )
