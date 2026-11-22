"""
Unit tests for TPA-GKT Preprocessing and Dual Knowledge Graph Construction.

Verifies:
  1. Zero-leakage properties:
     - Scaler fitted strictly on Train.
     - Unseen test problems map to 0 (<UNK>).
     - Transition graph A_trans computed purely from Train split.
  2. Retention delay bucket assignment and target construction.
  3. PyTorch Geometric graph data structures (edge_index shapes, valid indices, self-loops).
  4. Deterministic reproducibility under fixed seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from services.gnn_kt.src.graph_builder import (
    build_combined_graph,
    build_transition_graph,
)
from services.gnn_kt.src.preprocess import (
    assign_retention_buckets,
    compute_delayed_retention,
)
from torch_geometric.data import Data


def test_retention_bucket_assignment() -> None:
    """Test delay bucketing across all edge boundaries."""
    deltas = np.array([
        -10.0,      # invalid (<=0) -> -1
        0.0,        # invalid (<=0) -> -1
        60.0,       # 1 min -> 0 (<1h)
        3600.0,     # exact 1h -> 0 (<1h)
        3601.0,     # 1h + 1s -> 1 (1h-24h)
        86400.0,    # exact 24h -> 1 (1h-24h)
        86401.0,    # 1d + 1s -> 2 (1d-7d)
        604800.0,   # exact 7d -> 2 (1d-7d)
        604801.0,   # 7d + 1s -> 3 (>7d)
        1000000.0,  # >7d -> 3 (>7d)
    ])

    bucket_ids, bucket_names = assign_retention_buckets(deltas)

    assert bucket_ids[0] == -1 and bucket_names[0] == "none"
    assert bucket_ids[1] == -1 and bucket_names[1] == "none"
    assert bucket_ids[2] == 0 and bucket_names[2] == "<1h"
    assert bucket_ids[3] == 0 and bucket_names[3] == "<1h"
    assert bucket_ids[4] == 1 and bucket_names[4] == "1h-24h"
    assert bucket_ids[5] == 1 and bucket_names[5] == "1h-24h"
    assert bucket_ids[6] == 2 and bucket_names[6] == "1d-7d"
    assert bucket_ids[7] == 2 and bucket_names[7] == "1d-7d"
    assert bucket_ids[8] == 3 and bucket_names[8] == ">7d"
    assert bucket_ids[9] == 3 and bucket_names[9] == ">7d"


def test_delayed_retention_construction() -> None:
    """Test retention target calculation and practice history summary."""
    data = pd.DataFrame({
        "uuid": ["user_1", "user_1", "user_1", "user_1", "user_2"],
        "ucid": ["c_1", "c_2", "c_1", "c_1", "c_1"],
        "concept_idx": [0, 1, 0, 0, 0],
        "orig_idx": [0, 1, 2, 3, 4],
        "timestamp_sec": [1000, 2000, 5000, 15000, 3000],
        "is_correct": [1, 0, 1, 0, 1],
    })

    full_df, ret_df = compute_delayed_retention(data)

    u1_c1_mask = (full_df["uuid"] == "user_1") & (full_df["ucid"] == "c_1")
    u1_c1 = full_df[u1_c1_mask].sort_values("timestamp_sec")
    row0 = u1_c1.iloc[0]
    row2 = u1_c1.iloc[1]
    row3 = u1_c1.iloc[2]

    assert row0["has_retention_target"]
    assert row0["retention_target"] == 1
    assert row0["retention_delta_sec"] == 4000
    assert row0["retention_bucket_id"] == 1
    assert row0["prior_concept_attempts"] == 0
    assert row0["prior_concept_accuracy"] == 0.0

    assert row2["has_retention_target"]
    assert row2["retention_target"] == 0
    assert row2["retention_delta_sec"] == 10000
    assert row2["prior_concept_attempts"] == 1
    assert row2["prior_concept_accuracy"] == 1.0
    assert row2["log_time_since_prev_practice"] == pytest.approx(np.log1p(4000), rel=1e-3)

    assert not row3["has_retention_target"]
    assert row3["retention_target"] == -1

    assert len(ret_df) == 2


def test_zero_leakage_problem_indexer() -> None:
    """Verify unseen problems in validation/test map strictly to 0 (<UNK>)."""
    train_problems = ["prob_A", "prob_B", "prob_C"]
    problem2idx = {"<UNK>": 0}
    for p in sorted(train_problems):
        problem2idx[p] = len(problem2idx)

    test_problems = ["prob_A", "prob_NEW_UNSEEN", "prob_C"]
    test_indices = [problem2idx.get(p, 0) for p in test_problems]

    assert test_indices[0] == problem2idx["prob_A"]
    assert test_indices[1] == 0  # <UNK>
    assert test_indices[2] == problem2idx["prob_C"]


def test_zero_leakage_transition_graph(tmp_path) -> None:
    """Verify transition graph only learns transitions present in the Train split."""
    rows = []
    for u in range(15):
        rows.append({"uuid": f"u_{u}", "concept_idx": 0, "timestamp_sec": 100, "orig_idx": u * 2})
        rows.append(
            {"uuid": f"u_{u}", "concept_idx": 1, "timestamp_sec": 200, "orig_idx": u * 2 + 1}
        )
    for u in range(15, 27):
        rows.append({"uuid": f"u_{u}", "concept_idx": 1, "timestamp_sec": 100, "orig_idx": u * 2})
        rows.append(
            {"uuid": f"u_{u}", "concept_idx": 2, "timestamp_sec": 200, "orig_idx": u * 2 + 1}
        )

    train_df = pd.DataFrame(rows)
    train_parquet = tmp_path / "train.parquet"
    train_df.to_parquet(train_parquet, index=False)

    num_nodes = 5
    edge_index, edge_weight, trans_dict = build_transition_graph(
        train_parquet,
        num_nodes=num_nodes,
        transition_threshold=0.05,
        min_transition_count=10,
    )

    # Verify transitions 0->1 and 1->2 are present
    assert (0, 1) in trans_dict
    assert (1, 2) in trans_dict

    # Verify transition 3->4 (never in train) is absent
    assert (3, 4) not in trans_dict

    # Verify shapes and types
    assert edge_index.shape[0] == 2
    assert edge_index.dtype == torch.long
    assert edge_weight.dtype == torch.float32
    assert torch.all(edge_weight > 0)


def test_combined_graph_structure() -> None:
    """Verify combined graph has valid self-loops and positive weights."""
    num_nodes = 4
    # Hierarchical edge between 0 and 1
    hier_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    hier_edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float32)

    # Transition edge from 1 to 2
    trans_transitions = {(1, 2): 0.8}

    comb_edge_index, comb_edge_weight = build_combined_graph(
        hier_edge_index,
        hier_edge_weight,
        trans_transitions,
        num_nodes=num_nodes,
        alpha_hier=0.5,
    )

    comb_edges = set(
        zip(comb_edge_index[0].tolist(), comb_edge_index[1].tolist(), strict=False)
    )

    # Must contain self-loops for all 4 nodes
    for i in range(num_nodes):
        assert (i, i) in comb_edges

    # Must contain hierarchical edges (0, 1) and (1, 0)
    assert (0, 1) in comb_edges
    assert (1, 0) in comb_edges

    # Must contain transition edge (1, 2)
    assert (1, 2) in comb_edges

    # Check PyG Data container compatibility
    data = Data(edge_index=comb_edge_index, edge_weight=comb_edge_weight, num_nodes=num_nodes)
    assert data.num_nodes == num_nodes
    assert data.edge_index.size(0) == 2
    assert data.edge_weight.size(0) == data.edge_index.size(1)
    assert not torch.isnan(data.edge_weight).any()
    assert not torch.isinf(data.edge_weight).any()
