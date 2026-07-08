"""Deprecated compatibility wrapper for the TD3-GNN evaluator.

Use `evaluate_td3_gnn.py` for new experiments.
"""

from evaluate_td3_gnn import *  # noqa: F401,F403
from evaluate_td3_gnn import main


if __name__ == "__main__":
    main()
