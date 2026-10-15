import math
import numpy as np
from torch_geometric.data import Data


def PublicPST_GNN(env, *args):
    """
    Scale-agnostic PublicPST graph-state encoder for TD3 EV-GNN.

    The graph is pruned: only transformer/charger branches with a connected EV
    are included. The final action vector is reconstructed through
    data.action_mapper, which maps EV-node outputs back to EV2Gym action indexes.
    """

    ev_features = []
    cs_features = []
    tr_features = []

    env_features = [
        env.sim_date.weekday() / 7,
        math.sin(env.sim_date.hour / 24 * 2 * math.pi),
        math.cos(env.sim_date.hour / 24 * 2 * math.pi),
    ]

    if env.current_step < env.simulation_length:
        setpoint = env.power_setpoints[env.current_step]
    else:
        setpoint = 0

    previous_power_usage = env.current_power_usage[env.current_step - 1]
    env_features.append(setpoint)
    env_features.append(previous_power_usage)
    env_features = [env_features]

    node_features = [env_features]
    node_types = [0]
    node_names = ["env"]
    node_counter = 1

    ev_indexes = []
    cs_indexes = []
    tr_indexes = []
    env_indexes = [0]
    action_mapper = []

    edge_index_from = []
    edge_index_to = []

    port_counter = 0
    mapper = {}
    for cs in env.charging_stations:
        for port_id in range(cs.n_ports):
            mapper[f"Tr_{cs.connected_transformer}_CS_{cs.id}_EV_{port_id}"] = port_counter + port_id
        port_counter += cs.n_ports

    for tr in env.transformers:
        registered_tr = False
        tr_node_index = None

        for cs in env.charging_stations:
            if cs.connected_transformer != tr.id:
                continue

            registered_cs = False
            cs_node_index = None

            for ev in cs.evs_connected:
                if ev is None:
                    continue

                if not registered_cs:
                    if not registered_tr:
                        transformer_features = [tr.max_power[env.current_step], tr.id]
                        node_features.append(transformer_features)
                        tr_features.append(transformer_features)
                        tr_indexes.append(node_counter)
                        node_counter += 1
                        node_types.append(1)
                        node_names.append(f"Tr_{tr.id}")
                        tr_node_index = len(node_names) - 1

                        edge_index_from.extend([0, tr_node_index])
                        edge_index_to.extend([tr_node_index, 0])
                        registered_tr = True

                    charger_features = [
                        cs.min_charge_current,
                        cs.max_charge_current,
                        cs.n_ports,
                        cs.id,
                    ]
                    node_features.append(charger_features)
                    cs_features.append(charger_features)
                    cs_indexes.append(node_counter)
                    node_counter += 1
                    node_types.append(2)
                    node_names.append(f"Tr_{tr.id}_CS_{cs.id}")
                    cs_node_index = len(node_names) - 1

                    edge_index_from.extend([tr_node_index, cs_node_index])
                    edge_index_to.extend([cs_node_index, tr_node_index])
                    registered_cs = True

                ev_features_i = [
                    1 if ev.get_soc() == 1 else 0.5,
                    ev.total_energy_exchanged,
                    env.current_step - ev.time_of_arrival,
                    ev.id,
                    cs.id,
                    tr.id,
                ]
                node_features.append(ev_features_i)
                ev_features.append(ev_features_i)

                ev_indexes.append(node_counter)
                node_counter += 1
                node_types.append(3)

                action_key = f"Tr_{tr.id}_CS_{cs.id}_EV_{ev.id}"
                action_mapper.append(mapper[action_key])
                node_names.append(action_key)
                ev_node_index = len(node_names) - 1

                edge_index_from.extend([cs_node_index, ev_node_index])
                edge_index_to.extend([ev_node_index, cs_node_index])

    edge_index = np.array([edge_index_from, edge_index_to], dtype=int)

    return Data(
        ev_features=np.array(ev_features).reshape(-1, 6).astype(float),
        cs_features=np.array(cs_features).reshape(-1, 4).astype(float),
        tr_features=np.array(tr_features).reshape(-1, 2).astype(float),
        env_features=np.array(env_features).reshape(-1, 5).astype(float),
        edge_index=edge_index,
        node_types=np.array(node_types).astype(int),
        sample_node_length=[len(node_features)],
        action_mapper=action_mapper,
        ev_indexes=np.array(ev_indexes).astype(int),
        cs_indexes=np.array(cs_indexes).astype(int),
        tr_indexes=np.array(tr_indexes).astype(int),
        env_indexes=np.array(env_indexes).astype(int),
    )


# Expose node sizes immediately, rather than only after the first function call.
PublicPST_GNN.node_sizes = {"ev": 6, "cs": 4, "tr": 2, "env": 5}
