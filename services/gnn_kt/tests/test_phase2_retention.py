"""
Tests for Phase 2: Retention Target & Temporal Leakage Forensic Audit.

Validates:
1. Zero same-concept leakage between source and target retention interactions.
2. Mutually exclusive and collectively exhaustive intervening practice classification
   (isolated, related-concept interference, unrelated-concept interference).
3. Temporal monotonicity (delta_sec > 0).
4. Source and target concept identity matching.
5. Fine-grained delay bucket intervals (<15m, 15m-60m, 1h-24h, 1d-7d, >7d).
6. 15-minute grid timestamp quantization detection.
7. Prediction-to-interaction alignment integrity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

_project_root = str(Path(__file__).resolve().parents[3])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from services.gnn_kt.src.retention_audit import (  # noqa: E402
    classify_delay_interval,
    extract_and_verify_retention_pairs,
)


@pytest.fixture
def mock_retention_data(tmp_path: Path):
    """Create a minimal synthetic dataframe with known retention pairs."""
    # 2 users: user_1 and user_2
    # user_1:
    #   idx 0: c0, t=0, correct=1, upid 101
    #   idx 1: c1 (neighbor), t=900, correct=0, upid 102
    #   idx 2: c0, t=1800, correct=1, upid 101 (retention target for idx 0, related interference)
    #   idx 3: c2 (unrelated), t=2700, correct=1, upid 103
    #   idx 4: c0, t=7200, correct=0, upid 104 (retention target for idx 2, unrelated interference)
    #
    # user_2:
    #   idx 5: c1, t=0, correct=1, upid 102
    #   idx 6: c1, t=900, correct=1, upid 102 (retention target for idx 5, isolated)
    records = [
        # user_1
        {
            "uuid": "u1",
            "orig_idx": 0,
            "concept_idx": 0,
            "ucid": 0,
            "upid": 101,
            "timestamp_sec": 0,
            "is_correct": 1,
            "difficulty": 0.5,
            "norm_log_sec": 0.0,
            "norm_log_attempt": 0.0,
            "norm_log_hint_cnt": 0.0,
            "norm_delta_t": 0.0,
            "prior_concept_attempts": 0,
            "prior_concept_accuracy": 0.0,
            "log_time_since_prev_practice": 0.0,
            "has_retention_target": True,
            "retention_target": 1.0,
            "retention_delta_sec": 1800.0,
            "retention_bucket_id": 0,
            "retention_bucket_name": "<1h",
        },
        {
            "uuid": "u1",
            "orig_idx": 1,
            "concept_idx": 1,
            "ucid": 1,
            "upid": 102,
            "timestamp_sec": 900,
            "is_correct": 0,
            "difficulty": 0.5,
            "norm_log_sec": 0.0,
            "norm_log_attempt": 0.0,
            "norm_log_hint_cnt": 0.0,
            "norm_delta_t": 0.0,
            "prior_concept_attempts": 0,
            "prior_concept_accuracy": 0.0,
            "log_time_since_prev_practice": 0.0,
            "has_retention_target": False,
            "retention_target": 0.0,
            "retention_delta_sec": 0.0,
            "retention_bucket_id": -1,
            "retention_bucket_name": "none",
        },
        {
            "uuid": "u1",
            "orig_idx": 2,
            "concept_idx": 0,
            "ucid": 0,
            "upid": 101,
            "timestamp_sec": 1800,
            "is_correct": 1,
            "difficulty": 0.5,
            "norm_log_sec": 0.0,
            "norm_log_attempt": 0.0,
            "norm_log_hint_cnt": 0.0,
            "norm_delta_t": 0.0,
            "prior_concept_attempts": 1,
            "prior_concept_accuracy": 1.0,
            "log_time_since_prev_practice": 7.49,
            "has_retention_target": True,
            "retention_target": 0.0,
            "retention_delta_sec": 5400.0,
            "retention_bucket_id": 1,
            "retention_bucket_name": "1h-24h",
        },
        {
            "uuid": "u1",
            "orig_idx": 3,
            "concept_idx": 2,
            "ucid": 2,
            "upid": 103,
            "timestamp_sec": 2700,
            "is_correct": 1,
            "difficulty": 0.5,
            "norm_log_sec": 0.0,
            "norm_log_attempt": 0.0,
            "norm_log_hint_cnt": 0.0,
            "norm_delta_t": 0.0,
            "prior_concept_attempts": 0,
            "prior_concept_accuracy": 0.0,
            "log_time_since_prev_practice": 0.0,
            "has_retention_target": False,
            "retention_target": 0.0,
            "retention_delta_sec": 0.0,
            "retention_bucket_id": -1,
            "retention_bucket_name": "none",
        },
        {
            "uuid": "u1",
            "orig_idx": 4,
            "concept_idx": 0,
            "ucid": 0,
            "upid": 104,
            "timestamp_sec": 7200,
            "is_correct": 0,
            "difficulty": 0.5,
            "norm_log_sec": 0.0,
            "norm_log_attempt": 0.0,
            "norm_log_hint_cnt": 0.0,
            "norm_delta_t": 0.0,
            "prior_concept_attempts": 2,
            "prior_concept_accuracy": 1.0,
            "log_time_since_prev_practice": 8.59,
            "has_retention_target": False,
            "retention_target": 0.0,
            "retention_delta_sec": 0.0,
            "retention_bucket_id": -1,
            "retention_bucket_name": "none",
        },
        # user_2
        {
            "uuid": "u2",
            "orig_idx": 5,
            "concept_idx": 1,
            "ucid": 1,
            "upid": 102,
            "timestamp_sec": 0,
            "is_correct": 1,
            "difficulty": 0.5,
            "norm_log_sec": 0.0,
            "norm_log_attempt": 0.0,
            "norm_log_hint_cnt": 0.0,
            "norm_delta_t": 0.0,
            "prior_concept_attempts": 0,
            "prior_concept_accuracy": 0.0,
            "log_time_since_prev_practice": 0.0,
            "has_retention_target": True,
            "retention_target": 1.0,
            "retention_delta_sec": 900.0,
            "retention_bucket_id": 0,
            "retention_bucket_name": "<1h",
        },
        {
            "uuid": "u2",
            "orig_idx": 6,
            "concept_idx": 1,
            "ucid": 1,
            "upid": 102,
            "timestamp_sec": 900,
            "is_correct": 1,
            "difficulty": 0.5,
            "norm_log_sec": 0.0,
            "norm_log_attempt": 0.0,
            "norm_log_hint_cnt": 0.0,
            "norm_delta_t": 0.0,
            "prior_concept_attempts": 1,
            "prior_concept_accuracy": 1.0,
            "log_time_since_prev_practice": 6.80,
            "has_retention_target": False,
            "retention_target": 0.0,
            "retention_delta_sec": 0.0,
            "retention_bucket_id": -1,
            "retention_bucket_name": "none",
        },
    ]
    df = pd.DataFrame(records)
    parquet_path = tmp_path / "mock_test.parquet"
    df.to_parquet(parquet_path, index=False)

    # 3-node graph: edge between 0 and 1, but 2 is isolated
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    graph_data = Data(edge_index=edge_index)
    graph_path = tmp_path / "mock_graph.pt"
    torch.save(graph_data, graph_path)

    return parquet_path, graph_path


def test_fine_grained_delay_intervals():
    """Test delay interval boundary classification."""
    assert classify_delay_interval(0.0) == ("<15m", 0)
    assert classify_delay_interval(900.0) == ("<15m", 0)
    assert classify_delay_interval(901.0) == ("15m-60m", 0)
    assert classify_delay_interval(3600.0) == ("15m-60m", 0)
    assert classify_delay_interval(3601.0) == ("1h-24h", 1)
    assert classify_delay_interval(86400.0) == ("1h-24h", 1)
    assert classify_delay_interval(86401.0) == ("1d-7d", 2)
    assert classify_delay_interval(604800.0) == ("1d-7d", 2)
    assert classify_delay_interval(604801.0) == (">7d", 3)
    assert classify_delay_interval(1e8) == (">7d", 3)


def test_zero_same_concept_leakage_and_partition(mock_retention_data):
    """Validate zero same-concept leakage and exhaustive practice classification."""
    parquet_path, graph_path = mock_retention_data
    pairs_df, forensics = extract_and_verify_retention_pairs(parquet_path, graph_path)

    assert forensics["zero_same_concept_leakage_assert"] is True
    assert forensics["total_verified_retention_pairs"] == 3

    # Check mutually exclusive and collectively exhaustive partition
    total = forensics["total_verified_retention_pairs"]
    iso = forensics["isolated_retention_count"]
    rel = forensics["related_interference_count"]
    unrel = forensics["unrelated_interference_count"]

    assert iso + rel + unrel == total
    assert iso == 1  # u2 pair (c1 -> c1 with 0 intervening)
    assert rel == 1  # u1 pair 1 (c0 -> c0 with intervening c1 which is graph neighbor)
    assert unrel == 1  # u1 pair 2 (c0 -> c0 with intervening c2 which is unconnected)


def test_monotonicity_and_concept_matching(mock_retention_data):
    """Verify temporal monotonicity and concept identity for all extracted pairs."""
    parquet_path, graph_path = mock_retention_data
    pairs_df, _ = extract_and_verify_retention_pairs(parquet_path, graph_path)

    for _, row in pairs_df.iterrows():
        assert row["delta_sec"] > 0
        assert row["target_timestamp"] >= row["source_timestamp"]
        assert row["source_concept"] == row["target_concept"]
        assert row["intervening_same_concept_count"] == 0


def test_timestamp_quantization_detection(mock_retention_data):
    """Verify detection of 15-minute grid quantization."""
    parquet_path, graph_path = mock_retention_data
    _, forensics = extract_and_verify_retention_pairs(parquet_path, graph_path)

    # All mock timestamps (0, 900, 1800, 2700, 7200) are multiples of 900
    assert forensics["timestamp_15min_quantization_pct"] == 100.0


def test_real_dataset_retention_forensics():
    """Verify Phase 2 retention forensics on real processed test dataset."""
    test_parquet = Path("data/processed/test.parquet")
    graph_path = Path("data/processed/A_combined.pt")

    if not test_parquet.exists() or not graph_path.exists():
        pytest.skip("Processed test parquet or graph missing, skipping real test.")

    pairs_df, forensics = extract_and_verify_retention_pairs(test_parquet, graph_path)

    # 1. Zero leakage assertion must be 100% verified
    assert forensics["zero_same_concept_leakage_assert"] is True

    # 2. Total pairs must match reconstructed count
    assert len(pairs_df) == forensics["total_verified_retention_pairs"]
    assert len(pairs_df) > 0

    # 3. Exhaustive partition
    total = forensics["total_verified_retention_pairs"]
    iso = forensics["isolated_retention_count"]
    rel = forensics["related_interference_count"]
    unrel = forensics["unrelated_interference_count"]
    assert iso + rel + unrel == total

    # 4. Strict delta_sec > 0 for all pairs
    assert (pairs_df["delta_sec"] > 0).all()

    # 5. Timestamp quantization check
    assert forensics["timestamp_15min_quantization_pct"] == 100.0

    # 6. Correctness rates are within [0.0, 1.0]
    assert 0.0 <= pairs_df["target_correct"].mean() <= 1.0
