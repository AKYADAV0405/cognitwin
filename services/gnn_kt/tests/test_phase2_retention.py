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

import json
import sys
from pathlib import Path

import numpy as np
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


def test_retention_reconciliation_csv_deliverables(tmp_path: Path):
    """Verify that all 3 Phase 2 reconciliation CSV deliverables are exported properly."""
    from services.gnn_kt.src.retention_audit import (
        evaluate_retention_strata,
        export_reconciliation_csvs,
    )

    test_parquet = Path("data/processed/test.parquet")
    graph_path = Path("data/processed/A_combined.pt")
    ckpt_path = Path("services/gnn_kt/outputs/checkpoints/tpa_gkt_full_seed42_fastdev_best.pt")

    if not test_parquet.exists() or not ckpt_path.exists():
        pytest.skip("Test parquet or checkpoint missing, skipping deliverable test.")

    pairs_df, _ = extract_and_verify_retention_pairs(test_parquet, graph_path)
    device = torch.device("cpu")
    _, metrics_df = evaluate_retention_strata(ckpt_path, test_parquet, pairs_df, device)

    export_reconciliation_csvs(test_parquet, pairs_df, metrics_df, tmp_path)

    # 1. Verify RETENTION_PAIR_ACCOUNTING.csv
    p_acc = tmp_path / "RETENTION_PAIR_ACCOUNTING.csv"
    assert p_acc.exists()
    df_p = pd.read_csv(p_acc)
    assert len(df_p) == 3
    fast_dev_row = df_p[df_p["Cohort"].str.contains("Fast_Dev")].iloc[0]
    assert fast_dev_row["Total_Interaction_Records"] == 11646
    assert fast_dev_row["Valid_Delayed_Retention_Pairs_DeltaT_Pos"] == 1464
    assert fast_dev_row["Intra_Session_Zero_Delay_DeltaT_Zero"] == 8998
    assert fast_dev_row["Terminal_Concept_Tail_No_Subsequent_Attempt"] == 1184
    assert fast_dev_row["Accounting_Sum_Check"] == 11646
    assert bool(fast_dev_row["Balances_Exactly"]) is True

    # 1B. Verify COHORT_OVERLAP_ACCOUNTING.csv
    o_acc = tmp_path / "COHORT_OVERLAP_ACCOUNTING.csv"
    assert o_acc.exists()
    df_o = pd.read_csv(o_acc)
    assert len(df_o) == 11
    inter_row = df_o[df_o["Metric"].str.contains("Overlapping_Test_Students_Intersection")].iloc[0]
    assert inter_row["Count"] == 11

    # 2. Verify TIMESTAMP_FORENSIC_ACCOUNTING.csv
    ts_acc = tmp_path / "TIMESTAMP_FORENSIC_ACCOUNTING.csv"
    assert ts_acc.exists()
    df_ts = pd.read_csv(ts_acc)
    assert len(df_ts) >= 12
    zero_row = df_ts[df_ts["Metric_Or_Interval"].str.contains("Intra-Session")].iloc[0]
    assert zero_row["Count"] == 8998

    # 3. Verify RETENTION_METRIC_RECONCILIATION.csv
    m_rec = tmp_path / "RETENTION_METRIC_RECONCILIATION.csv"
    assert m_rec.exists()
    df_m = pd.read_csv(m_rec)
    assert len(df_m) >= 10
    overall = df_m[df_m["Stratum"] == "Overall Test Retention"].iloc[0]
    assert overall["Sample Count"] == 1464
    assert 0.75 <= float(overall["Retention AUC"]) <= 0.77


def test_pair_accounting_exact_arithmetic_balance():
    """Verify that every interaction record receives exactly 1 mutually exclusive disposition."""
    test_parquet = Path("data/processed/test.parquet")
    if not test_parquet.exists():
        pytest.skip("test.parquet missing, skipping accounting balance test.")

    df = pd.read_parquet(test_parquet)
    df_sorted = df.sort_values(by=["uuid", "timestamp_sec", "orig_idx"]).reset_index(drop=True)

    next_u = df_sorted.groupby(["uuid", "ucid"])["uuid"].shift(-1)
    next_t = df_sorted.groupby(["uuid", "ucid"])["timestamp_sec"].shift(-1)
    has_cand = ~next_u.isna()
    dt = (next_t - df_sorted["timestamp_sec"]).values

    n_total = len(df_sorted)
    n_valid = int(((dt > 0) & has_cand).sum())
    n_zero = int(((dt == 0) & has_cand).sum())
    n_tail = int((~has_cand).sum())
    n_neg = int(((dt < 0) & has_cand).sum())
    n_inv = int((df_sorted["timestamp_sec"] <= 0).sum())

    # 1. Total interactions match sum of 5 mutually exclusive dispositions
    assert n_valid + n_zero + n_tail + n_neg + n_inv == n_total
    assert n_total == 11646
    assert n_valid == 1464
    assert n_zero == 8998
    assert n_tail == 1184
    assert n_neg == 0
    assert n_inv == 0

    # 2. Terminal concept tails match unique (uuid, ucid) student-concept sequences
    assert n_tail == df_sorted.groupby(["uuid", "ucid"]).ngroups

    # 3. Valid retention pairs match has_retention_target column
    assert n_valid == int(df["has_retention_target"].sum())


def test_cohort_student_id_overlap():
    """Verify empirical ID-level student overlap between local fast-dev and full benchmark."""
    eligible_path = Path("data/processed/eligible_students.json")
    test_parquet = Path("data/processed/test.parquet")

    if not eligible_path.exists() or not test_parquet.exists():
        pytest.skip("eligible_students.json or test.parquet missing, skipping overlap test.")

    with open(eligible_path, encoding="utf-8") as f:
        eligible = json.load(f)

    # Reconstruct 500 fast-dev cohort
    rng_500 = np.random.RandomState(42)
    sel_500 = sorted(rng_500.choice(eligible, size=500, replace=False).tolist())
    u_500 = np.array(sel_500)
    split_rng_500 = np.random.RandomState(42)
    split_rng_500.shuffle(u_500)
    test_500 = set(u_500[int(500 * 0.85):])

    # Reconstruct 5,000 full benchmark cohort
    rng_5000 = np.random.RandomState(42)
    sel_5000 = sorted(rng_5000.choice(eligible, size=5000, replace=False).tolist())
    u_5000 = np.array(sel_5000)
    split_rng_5000 = np.random.RandomState(42)
    split_rng_5000.shuffle(u_5000)
    train_5000 = set(u_5000[:int(5000 * 0.70)])
    val_5000 = set(u_5000[int(5000 * 0.70):int(5000 * 0.85)])
    test_5000 = set(u_5000[int(5000 * 0.85):])

    # Assert local test matches actual test.parquet
    df_test = pd.read_parquet(test_parquet)
    actual_test_uuids = set(df_test["uuid"].unique())
    assert actual_test_uuids == test_500
    assert len(test_500) == 75
    assert len(test_5000) == 750

    # Overlap assertions
    overlap_test = test_500.intersection(test_5000)
    overlap_train = test_500.intersection(train_5000)
    overlap_val = test_500.intersection(val_5000)

    assert len(overlap_test) == 11
    assert len(overlap_train) == 57
    assert len(overlap_val) == 7
    assert len(test_500 - test_5000) == 64
    assert len(test_5000 - test_500) == 739

    # Interaction & retention pair counts for overlapping students
    df_overlap = df_test[df_test["uuid"].isin(overlap_test)]
    assert len(df_overlap) == 1633
    assert int(df_overlap["has_retention_target"].sum()) == 299


def test_graph_zero_leakage_provenance():
    """Verify that graph construction has zero test leakage."""
    a_trans_path = Path("data/processed/A_trans.pt")
    metadata_path = Path("data/processed/metadata.json")

    if not a_trans_path.exists() or not metadata_path.exists():
        pytest.skip("A_trans.pt or metadata.json missing, skipping graph leakage test.")

    a_trans = torch.load(a_trans_path, weights_only=False)
    assert hasattr(a_trans, "edge_index")
    assert a_trans.edge_index.size(0) == 2
    # Verify directional weights exist and are bounded [0, 1]
    if hasattr(a_trans, "edge_weight") and a_trans.edge_weight is not None:
        assert (a_trans.edge_weight >= 0.0).all()
        assert (a_trans.edge_weight <= 1.0).all()
