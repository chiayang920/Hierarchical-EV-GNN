from TD3.TD3_HierarchicalActionGNN import TD3_HierarchicalActionGNN
import train_RL_GNN_25cp as td3_25cp_training


# ponytail: reuse the bounded 25 CP training loop; split it only if the loops diverge.
td3_25cp_training.TD3_ActionGNN = TD3_HierarchicalActionGNN


if __name__ == "__main__":
    td3_25cp_training.main()
