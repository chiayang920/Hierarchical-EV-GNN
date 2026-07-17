from pathlib import Path
import csv
import importlib
import json
import statistics
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


profiler = importlib.import_module("profiling.profile_hierarchical_actor_breakdown")


def build_active_state() -> Data:
    return Data(
        ev_features=np.array(
            [
                [0.5, 4.0, 1.0, 0.0, 0.0, 0.0],
                [0.5, 6.0, 2.0, 1.0, 1.0, 1.0],
            ],
            dtype=float,
        ),
        cs_features=np.array(
            [
                [0.0, 32.0, 1.0, 0.0],
                [0.0, 32.0, 1.0, 1.0],
            ],
            dtype=float,
        ),
        tr_features=np.array(
            [
                [100.0, 0.0],
                [100.0, 1.0],
            ],
            dtype=float,
        ),
        env_features=np.array([[0.0, 0.0, 1.0, 10.0, 0.0]], dtype=float),
        edge_index=np.array(
            [
                [0, 1, 1, 2, 2, 3, 0, 4, 4, 5, 5, 6],
                [1, 0, 2, 1, 3, 2, 4, 0, 5, 4, 6, 5],
            ],
            dtype=np.int64,
        ),
        node_types=np.array([0, 1, 2, 3, 1, 2, 3], dtype=int),
        sample_node_length=[7],
        action_mapper=[0, 1],
        ev_indexes=np.array([3, 6], dtype=int),
        cs_indexes=np.array([2, 5], dtype=int),
        tr_indexes=np.array([1, 4], dtype=int),
        env_indexes=np.array([0], dtype=int),
    )


def build_no_active_ev_state() -> Data:
    return Data(
        ev_features=np.empty((0, 6), dtype=float),
        cs_features=np.empty((0, 4), dtype=float),
        tr_features=np.empty((0, 2), dtype=float),
        env_features=np.array([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=float),
        edge_index=np.empty((2, 0), dtype=np.int64),
        node_types=np.array([0], dtype=int),
        sample_node_length=[1],
        action_mapper=[],
        ev_indexes=np.array([], dtype=int),
        cs_indexes=np.array([], dtype=int),
        tr_indexes=np.array([], dtype=int),
        env_indexes=np.array([0], dtype=int),
    )


def build_manual_batched_state() -> Data:
    from utils.replay_buffer_actiongnn import _batch_graphs

    return _batch_graphs([build_active_state(), build_active_state()], device="cpu")


def build_selector_state(active_ev_count: int, total_nodes: int) -> Data:
    return Data(
        node_types=np.arange(total_nodes, dtype=int),
        sample_node_length=[total_nodes],
        ev_indexes=np.arange(active_ev_count, dtype=int),
        cs_indexes=np.array([], dtype=int),
        tr_indexes=np.array([], dtype=int),
        env_indexes=np.array([0], dtype=int),
    )


class FakeStepSequence:
    def __init__(self, states, done_flags):
        self.states = states
        self.done_flags = done_flags
        self.step_index = 0

    def step(self, state):
        assert state is self.states[self.step_index]
        done = self.done_flags[self.step_index]
        next_state_index = min(self.step_index + 1, len(self.states) - 1)
        self.step_index += 1
        return self.states[next_state_index], 0.0, done, {}


def run_fake_collection(states, done_flags, target_replay_size, max_collection_steps=None):
    sequence = FakeStepSequence(states=states, done_flags=done_flags)
    added_transitions = []

    def add_active_transition(state, next_state, reward, done):
        added_transitions.append((state, next_state, reward, done))

    def reset_episode(episode_index):
        raise AssertionError(f"Unexpected reset for episode {episode_index}")

    result = profiler.collect_until_threshold_episode_complete(
        initial_state=states[0],
        target_replay_size=target_replay_size,
        max_collection_steps=max_collection_steps or len(done_flags),
        step_state=sequence.step,
        reset_episode=reset_episode,
        add_active_transition=add_active_transition,
        log_every=0,
    )
    return result, added_transitions, sequence


def make_actiongnn_actor():
    from TD3.TD3_ActionGNN_Controlled import Actor
    from utils.state_public_pst_gnn import PublicPST_GNN

    return Actor(
        max_action=1.0,
        fx_node_sizes=PublicPST_GNN.node_sizes,
        feature_dim=8,
        GNN_hidden_dim=16,
        num_gcn_layers=3,
        discrete_actions=1,
        device=torch.device("cpu"),
    )


def make_hierarchical_actor():
    from TD3.TD3_HierarchicalActionGNN import Actor
    from utils.state_public_pst_gnn import PublicPST_GNN

    return Actor(
        max_action=1.0,
        fx_node_sizes=PublicPST_GNN.node_sizes,
        feature_dim=8,
        GNN_hidden_dim=16,
        num_gcn_layers=3,
        discrete_actions=1,
        device=torch.device("cpu"),
    )


def make_timing_record(
    algorithm: str,
    batch_size: int,
    schedule_index: int,
    active_ev_count: int,
    total_nodes: int,
):
    return profiler.TimingRecord(
        run_name="unit",
        config="config_files/PublicPST_25cp.yaml",
        scale="PublicPST_25cp",
        algorithm=algorithm,
        batch_size=batch_size,
        repetition=0,
        status="measured",
        phase=f"{algorithm}_actor_forward",
        elapsed_seconds=0.01,
        total_nodes=total_nodes,
        active_ev_count=active_ev_count,
        charger_count=active_ev_count,
        transformer_count=1,
        num_graphs=1,
        cpu_thread_count=1,
        device="cpu",
        schedule_index=schedule_index,
        algorithm_order_position=0,
    )


def test_cli_parsing_accepts_required_breakdown_options(tmp_path):
    args = profiler.parse_args(
        [
            "--config",
            "config_files/PublicPST_25cp.yaml",
            "--output_dir",
            str(tmp_path),
            "--warmup_repetitions",
            "0",
            "--measured_repetitions",
            "1",
            "--batch_sizes",
            "1",
            "64",
        ]
    )

    assert args.config == "config_files/PublicPST_25cp.yaml"
    assert args.output_dir == str(tmp_path)
    assert args.warmup_repetitions == 0
    assert args.measured_repetitions == 1
    assert args.batch_sizes == [1, 64]


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--measured_repetitions", "0"],
        ["--warmup_repetitions", "-1"],
        ["--batch_sizes", "0"],
    ],
)
def test_invalid_repetition_counts_and_batch_sizes_fail_clearly(tmp_path, extra_args):
    base_args = [
        "--config",
        "config_files/PublicPST_25cp.yaml",
        "--output_dir",
        str(tmp_path),
    ]

    with pytest.raises(SystemExit):
        profiler.parse_args(base_args + extra_args)


def test_summary_statistic_calculation_uses_measured_rows_only():
    records = [
        profiler.TimingRecord(
            run_name="unit",
            config="config_files/PublicPST_25cp.yaml",
            scale="PublicPST_25cp",
            algorithm="hierarchical",
            batch_size=1,
            repetition=0,
            status="warmup",
            phase="hierarchical_actor_forward",
            elapsed_seconds=100.0,
            total_nodes=7,
            active_ev_count=2,
            charger_count=2,
            transformer_count=2,
            num_graphs=1,
            cpu_thread_count=1,
            device="cpu",
        ),
        profiler.TimingRecord(
            run_name="unit",
            config="config_files/PublicPST_25cp.yaml",
            scale="PublicPST_25cp",
            algorithm="hierarchical",
            batch_size=1,
            repetition=0,
            status="measured",
            phase="hierarchical_actor_forward",
            elapsed_seconds=1.0,
            total_nodes=7,
            active_ev_count=2,
            charger_count=2,
            transformer_count=2,
            num_graphs=1,
            cpu_thread_count=1,
            device="cpu",
        ),
        profiler.TimingRecord(
            run_name="unit",
            config="config_files/PublicPST_25cp.yaml",
            scale="PublicPST_25cp",
            algorithm="hierarchical",
            batch_size=1,
            repetition=1,
            status="measured",
            phase="hierarchical_actor_forward",
            elapsed_seconds=3.0,
            total_nodes=7,
            active_ev_count=2,
            charger_count=2,
            transformer_count=2,
            num_graphs=1,
            cpu_thread_count=1,
            device="cpu",
        ),
    ]

    summary = profiler.build_json_summary(records, metadata={"run_name": "unit"})
    stats = summary["phases"]["hierarchical"]["batch_1"]["hierarchical_actor_forward"]

    assert stats["count"] == 2
    assert stats["total"] == pytest.approx(4.0)
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["median"] == pytest.approx(2.0)
    assert stats["stdev"] == pytest.approx(statistics.stdev([1.0, 3.0]))
    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(3.0)
    assert stats["p95"] == pytest.approx(3.0)


def test_required_csv_and_json_schema(tmp_path):
    required_csv_columns = {
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
    }
    assert required_csv_columns.issubset(set(profiler.CSV_FIELDNAMES))

    record = profiler.TimingRecord(
        run_name="unit",
        config="config_files/PublicPST_25cp.yaml",
        scale="PublicPST_25cp",
        algorithm="actiongnn",
        batch_size=1,
        repetition=0,
        status="measured",
        phase="actiongnn_actor_forward",
        elapsed_seconds=0.01,
        total_nodes=7,
        active_ev_count=2,
        charger_count=2,
        transformer_count=2,
        num_graphs=1,
        cpu_thread_count=1,
        device="cpu",
    )
    summary = profiler.build_json_summary([record], metadata={"run_name": "unit"})

    csv_path, json_path = profiler.write_outputs(
        records=[record],
        summary=summary,
        output_dir=tmp_path,
        run_name="unit",
    )

    with csv_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    with json_path.open() as json_file:
        loaded_summary = json.load(json_file)

    assert rows[0]["phase"] == "actiongnn_actor_forward"
    assert required_csv_columns.issubset(rows[0].keys())
    assert {"count", "total", "mean", "median", "stdev", "min", "max", "p95"}.issubset(
        loaded_summary["phases"]["actiongnn"]["batch_1"]["actiongnn_actor_forward"].keys()
    )


def test_algorithm_order_alternates_by_schedule_index():
    assert profiler.algorithm_order_for_schedule_index(0) == ("actiongnn", "hierarchical")
    assert profiler.algorithm_order_for_schedule_index(1) == ("hierarchical", "actiongnn")
    assert profiler.algorithm_order_for_schedule_index(2) == ("actiongnn", "hierarchical")

    schedule = profiler.repetition_schedule(
        SimpleNamespace(warmup_repetitions=1, measured_repetitions=2)
    )
    assert schedule == [
        (0, "warmup", 0),
        (1, "measured", 0),
        (2, "measured", 1),
    ]


def test_timing_wrapper_does_not_change_actor_output():
    torch.manual_seed(0)
    state = build_active_state()
    actor = make_hierarchical_actor()

    expected_output = actor(state).detach()
    timed_output, elapsed_seconds = profiler.time_callable(lambda: actor(state))

    assert elapsed_seconds >= 0.0
    assert torch.allclose(timed_output.detach(), expected_output, atol=0.0, rtol=0.0)


def test_hierarchical_actor_output_shape_and_non_ev_rows_remain_zero():
    torch.manual_seed(1)
    state = build_active_state()
    actor = make_hierarchical_actor()

    full_node_action = actor(state)

    assert full_node_action.shape == (int(sum(state.sample_node_length)), 1)
    profiler.assert_non_ev_action_rows_zero(full_node_action, state)


def test_complete_forward_matches_decomposed_forward_for_one_active_graph():
    torch.manual_seed(4)
    state = build_active_state()
    actor = make_hierarchical_actor()

    complete_output = actor(state)
    decomposed_output, phase_timings = profiler.measure_hierarchical_forward_breakdown(actor, state)

    profiler.assert_hierarchical_forward_matches_breakdown(complete_output, decomposed_output)
    assert phase_timings["hierarchical_compose_full_node_action"] >= 0.0


def test_complete_forward_matches_decomposed_forward_for_batched_graphs():
    torch.manual_seed(5)
    state = build_manual_batched_state()
    actor = make_hierarchical_actor()

    complete_output = actor(state)
    decomposed_output, phase_timings = profiler.measure_hierarchical_forward_breakdown(actor, state)

    assert state.sample_node_length == [7, 7]
    profiler.assert_hierarchical_forward_matches_breakdown(complete_output, decomposed_output)
    assert phase_timings["hierarchical_encode_nodes"] >= 0.0


def test_decomposed_forward_equivalence_failure_is_clear():
    complete_output = torch.zeros((2, 1), dtype=torch.float32)
    decomposed_output = torch.ones((2, 1), dtype=torch.float32)

    with pytest.raises(AssertionError, match="Decomposed hierarchical actor forward"):
        profiler.assert_hierarchical_forward_matches_breakdown(complete_output, decomposed_output)


def test_no_active_ev_state_is_handled_by_breakdown():
    state = build_no_active_ev_state()
    actor = make_hierarchical_actor()

    complete_output = actor(state)
    full_node_action, phase_timings = profiler.measure_hierarchical_forward_breakdown(actor, state)

    assert full_node_action.shape == (1, 1)
    assert torch.allclose(full_node_action, torch.zeros((1, 1)))
    profiler.assert_hierarchical_forward_matches_breakdown(complete_output, full_node_action)
    assert phase_timings["hierarchical_prepare_features"] >= 0.0
    assert phase_timings["hierarchical_encode_nodes"] >= 0.0
    assert phase_timings["hierarchical_compose_full_node_action"] >= 0.0


def test_representative_state_selection_prefers_more_active_evs():
    low_ev_state = build_selector_state(active_ev_count=1, total_nodes=20)
    high_ev_state = build_selector_state(active_ev_count=3, total_nodes=5)

    selected_state = profiler.select_representative_single_state([low_ev_state, high_ev_state])

    assert selected_state is high_ev_state


def test_representative_state_selection_prefers_more_nodes_when_ev_counts_tie():
    small_state = build_selector_state(active_ev_count=2, total_nodes=5)
    large_state = build_selector_state(active_ev_count=2, total_nodes=9)

    selected_state = profiler.select_representative_single_state([small_state, large_state])

    assert selected_state is large_state


def test_representative_state_selection_keeps_earliest_when_counts_tie():
    first_state = build_selector_state(active_ev_count=2, total_nodes=7)
    second_state = build_selector_state(active_ev_count=2, total_nodes=7)

    selected_state = profiler.select_representative_single_state([first_state, second_state])

    assert selected_state is first_state


def test_threshold_does_not_finish_collection_until_episode_done():
    assert profiler.should_finish_collection(replay_threshold_reached=True, episode_done=False) is False
    assert profiler.should_finish_collection(replay_threshold_reached=True, episode_done=True) is True

    states = [
        build_selector_state(active_ev_count=1, total_nodes=3),
        build_selector_state(active_ev_count=2, total_nodes=4),
    ]
    result, added_transitions, sequence = run_fake_collection(
        states=states,
        done_flags=[False, True],
        target_replay_size=1,
    )

    assert sequence.step_index == 2
    assert len(added_transitions) == 2
    assert result["collection_steps_completed"] == 2
    assert result["collection_termination_reason"] == "completed_threshold_episode"


def test_threshold_episode_completion_terminates_collection_immediately():
    states = [
        build_selector_state(active_ev_count=1, total_nodes=3),
        build_selector_state(active_ev_count=2, total_nodes=4),
        build_selector_state(active_ev_count=10, total_nodes=20),
    ]
    result, added_transitions, sequence = run_fake_collection(
        states=states,
        done_flags=[False, True, False],
        target_replay_size=1,
        max_collection_steps=3,
    )

    assert sequence.step_index == 2
    assert len(added_transitions) == 2
    assert result["collection_steps_completed"] == 2
    assert result["representative_single_state"] is states[1]


def test_post_threshold_higher_load_state_can_become_representative():
    states = [
        build_selector_state(active_ev_count=1, total_nodes=3),
        build_selector_state(active_ev_count=5, total_nodes=7),
        build_selector_state(active_ev_count=3, total_nodes=5),
    ]
    result, added_transitions, sequence = run_fake_collection(
        states=states,
        done_flags=[False, True, False],
        target_replay_size=1,
        max_collection_steps=3,
    )

    assert sequence.step_index == 2
    assert len(added_transitions) == 2
    assert result["representative_single_state"] is states[1]
    assert result["representative_state_collection_step"] == 2
    assert result["representative_state_episode_index"] == 0
    assert result["representative_single_state_counts"]["active_ev_count"] == 5


def test_collection_metadata_records_threshold_and_representative_steps():
    states = [
        build_selector_state(active_ev_count=1, total_nodes=3),
        build_selector_state(active_ev_count=2, total_nodes=4),
        build_selector_state(active_ev_count=4, total_nodes=9),
    ]
    result, added_transitions, sequence = run_fake_collection(
        states=states,
        done_flags=[False, False, True],
        target_replay_size=2,
    )

    assert sequence.step_index == 3
    assert len(added_transitions) == 3
    assert result["collection_steps_completed"] == 3
    assert result["collected_active_transition_count"] == 3
    assert result["completed_collection_episodes"] == 1
    assert result["replay_threshold_reached_at_step"] == 2
    assert result["replay_threshold_episode_index"] == 0
    assert result["representative_state_collection_step"] == 3
    assert result["representative_state_episode_index"] == 0
    assert result["representative_single_state_counts"] == {
        "total_nodes": 9,
        "active_ev_count": 4,
        "charger_count": 0,
        "transformer_count": 0,
        "num_graphs": 1,
    }
    assert result["collection_termination_reason"] == "completed_threshold_episode"


def test_collection_fails_if_threshold_episode_does_not_finish_before_max_steps():
    states = [
        build_selector_state(active_ev_count=1, total_nodes=3),
        build_selector_state(active_ev_count=2, total_nodes=4),
    ]
    sequence = FakeStepSequence(states=states, done_flags=[False, False])

    with pytest.raises(RuntimeError, match="threshold episode did not terminate"):
        profiler.collect_until_threshold_episode_complete(
            initial_state=states[0],
            target_replay_size=1,
            max_collection_steps=2,
            step_state=sequence.step,
            reset_episode=lambda episode_index: states[0],
            add_active_transition=lambda state, next_state, reward, done: None,
            log_every=0,
        )


def test_matched_graph_counts_accepts_identical_algorithm_counts():
    records = [
        make_timing_record("actiongnn", batch_size=64, schedule_index=0, active_ev_count=12, total_nodes=20),
        make_timing_record("hierarchical", batch_size=64, schedule_index=0, active_ev_count=12, total_nodes=20),
    ]

    profiler.assert_matched_algorithm_graph_counts(records)


def test_matched_graph_counts_rejects_different_algorithm_counts():
    records = [
        make_timing_record("actiongnn", batch_size=64, schedule_index=0, active_ev_count=12, total_nodes=20),
        make_timing_record("hierarchical", batch_size=64, schedule_index=0, active_ev_count=10, total_nodes=20),
    ]

    with pytest.raises(AssertionError, match="Mismatched graph counts"):
        profiler.assert_matched_algorithm_graph_counts(records)


@pytest.mark.parametrize(
    ("algorithm", "actor_factory"),
    [
        ("actiongnn", make_actiongnn_actor),
        ("hierarchical", make_hierarchical_actor),
    ],
)
def test_diagnostic_backward_produces_finite_gradients(algorithm, actor_factory):
    torch.manual_seed(2)
    state = build_active_state()
    actor = actor_factory()

    output, loss = profiler.run_actor_forward_backward(actor, state, algorithm)

    assert torch.isfinite(output).all()
    assert torch.isfinite(loss).all()
    profiler.assert_finite_gradients(actor)


def test_diagnostic_backward_does_not_modify_policy_parameters_or_write_checkpoints(tmp_path):
    torch.manual_seed(3)
    state = build_active_state()
    actor = make_hierarchical_actor()
    before_parameters = profiler.snapshot_parameters(actor)

    profiler.run_actor_forward_backward(actor, state, "hierarchical")

    assert profiler.parameters_match_snapshot(actor, before_parameters)

    record = profiler.TimingRecord(
        run_name="unit",
        config="config_files/PublicPST_25cp.yaml",
        scale="PublicPST_25cp",
        algorithm="hierarchical",
        batch_size=1,
        repetition=0,
        status="measured",
        phase="hierarchical_actor_forward_backward",
        elapsed_seconds=0.02,
        total_nodes=7,
        active_ev_count=2,
        charger_count=2,
        transformer_count=2,
        num_graphs=1,
        cpu_thread_count=1,
        device="cpu",
    )
    summary = profiler.build_json_summary([record], metadata={"run_name": "unit"})
    profiler.write_outputs([record], summary, tmp_path, "unit")

    output_files = {path.name for path in tmp_path.iterdir()}
    assert output_files == {
        "unit_hierarchical_actor_breakdown_steps.csv",
        "unit_hierarchical_actor_breakdown_summary.json",
    }
