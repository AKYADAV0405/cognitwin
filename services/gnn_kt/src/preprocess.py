"""
Temporal Prerequisite-Aware Graph Knowledge Tracing (TPA-GKT)
Preprocess Module: Chunked Zero-Leakage Preprocessing & Retention Target Construction

Dataset: Junyi Academy 2020 (Log_Problem.csv, Info_Content.csv)
Paper Scope: Section IV.C (Empirical Telemetry) & Section IV.G (Delayed-Retention Evaluation)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler


def resolve_path(path_str: str, base_dirs: list[Path]) -> Path:
    """Resolve a path against candidate base directories."""
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    for base in base_dirs:
        candidate = (base / p).resolve()
        if candidate.exists():
            return candidate
    return (base_dirs[0] / p).resolve()


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and parse YAML configuration file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def assign_retention_buckets(delta_sec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized assignment of retention delay buckets.

    Buckets:
      - bucket 0: '<1h'     (0 < delta <= 3600)
      - bucket 1: '1h-24h'  (3600 < delta <= 86400)
      - bucket 2: '1d-7d'   (86400 < delta <= 604800)
      - bucket 3: '>7d'     (delta > 604800)
    """
    conditions = [
        (delta_sec > 0) & (delta_sec <= 3600),
        (delta_sec > 3600) & (delta_sec <= 86400),
        (delta_sec > 86400) & (delta_sec <= 604800),
        delta_sec > 604800,
    ]
    bucket_ids = np.select(conditions, [0, 1, 2, 3], default=-1).astype(np.int64)
    bucket_names = np.select(conditions, ["<1h", "1h-24h", "1d-7d", ">7d"], default="none")
    return bucket_ids, bucket_names


def compute_delayed_retention(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct delayed-retention prediction targets (Paper 1 Section IV.G).

    For every interaction (u, c, t), where student u later attempts concept c at t' > t
    with delta = tau_{t'} - tau_t > 0, compute (u, c, t, delta, y_F = r_{t'})
    plus bucket label and concept practice summary v_{u,c,t}.
    """
    df = df.copy()

    # Per-student, per-concept chronological practice summary
    df["concept_attempt_idx"] = df.groupby(["uuid", "ucid"]).cumcount()
    cum_correct = df.groupby(["uuid", "ucid"])["is_correct"].cumsum() - df["is_correct"]
    df["prior_concept_attempts"] = df["concept_attempt_idx"].astype(np.int64)
    df["prior_concept_accuracy"] = np.where(
        df["prior_concept_attempts"] > 0,
        (cum_correct / np.maximum(1, df["prior_concept_attempts"])).astype(np.float32),
        0.0,
    )

    prev_time = df.groupby(["uuid", "ucid"])["timestamp_sec"].shift(1)
    time_since_prev = df["timestamp_sec"] - prev_time
    df["log_time_since_prev_practice"] = np.where(
        ~prev_time.isna(),
        np.log1p(np.maximum(0.0, time_since_prev)).astype(np.float32),
        0.0,
    )

    # Next attempt targets on the same concept
    next_time = df.groupby(["uuid", "ucid"])["timestamp_sec"].shift(-1)
    next_correct = df.groupby(["uuid", "ucid"])["is_correct"].shift(-1)

    delta_sec = (next_time - df["timestamp_sec"]).values
    has_retention = (~next_correct.isna().values) & (delta_sec > 0)

    bucket_ids, bucket_names = assign_retention_buckets(delta_sec)

    df["has_retention_target"] = has_retention
    df["retention_target"] = np.where(has_retention, next_correct.fillna(0).astype(np.int64), -1)
    df["retention_delta_sec"] = np.where(has_retention, delta_sec, -1.0)
    df["retention_bucket_id"] = np.where(has_retention, bucket_ids, -1)
    df["retention_bucket_name"] = np.where(has_retention, bucket_names, "none")

    # Clean up temporary helper column
    df = df.drop(columns=["concept_attempt_idx"])

    # Standalone retention evaluation table
    retention_cols = [
        "uuid",
        "ucid",
        "concept_idx",
        "orig_idx",
        "timestamp_sec",
        "retention_delta_sec",
        "retention_bucket_id",
        "retention_bucket_name",
        "retention_target",
        "prior_concept_attempts",
        "prior_concept_accuracy",
        "log_time_since_prev_practice",
    ]
    retention_df = df[df["has_retention_target"]][retention_cols].copy().reset_index(drop=True)

    return df, retention_df


def run_preprocessing(
    config_path: str = "services/gnn_kt/config.yaml",
    fast_dev_run: bool = False,
    override_seed: int | None = None,
) -> dict[str, Any]:
    """Execute chunked streaming preprocessing and zero-leakage dataset generation."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent.parent

    config_candidates = [repo_root, script_dir.parent, Path.cwd()]
    cfg_file = resolve_path(config_path, config_candidates)
    cfg = load_config(cfg_file)

    seed = override_seed if override_seed is not None else cfg["seeds"][0]

    # Resolve Data Directories
    data_raw_dir = resolve_path(cfg["data_raw_dir"], [repo_root, script_dir.parent, Path.cwd()])
    data_proc_dir = resolve_path(
        cfg["data_processed_dir"], [repo_root, script_dir.parent, Path.cwd()]
    )
    data_proc_dir.mkdir(parents=True, exist_ok=True)

    log_problem_csv = data_raw_dir / "Log_Problem.csv"
    info_content_csv = data_raw_dir / "Info_Content.csv"

    if not log_problem_csv.exists() or not info_content_csv.exists():
        raise FileNotFoundError(
            f"Missing required raw CSVs in {data_raw_dir}: "
            f"Log_Problem={log_problem_csv.exists()}, Info_Content={info_content_csv.exists()}"
        )

    # Configuration Parameters
    top_n_concepts = int(cfg["top_n_concepts"])
    min_interactions = int(cfg["min_interactions"])
    max_interactions = int(cfg.get("max_interactions", 1000))
    chunksize = int(cfg.get("chunksize", 1000000))
    train_ratio = float(cfg.get("train_ratio", 0.70))
    val_ratio = float(cfg.get("val_ratio", 0.15))

    target_max_students = (
        int(cfg["fast_dev_max_students"]) if fast_dev_run else int(cfg["max_students"])
    )

    print("=" * 78)
    print("TPA-GKT PREPROCESSING: ZERO-LEAKAGE CHUNKED STREAMING")
    print(f"Data Raw Dir     : {data_raw_dir}")
    print(f"Data Processed   : {data_proc_dir}")
    print(f"Fast Dev Run     : {fast_dev_run} (Target Students: {target_max_students:,})")
    print(f"Top Concepts (N) : {top_n_concepts}")
    print(f"Min Interactions : {min_interactions}")
    print(f"Random Seed      : {seed}")
    print("=" * 78)

    # --------------------------------------------------------------------------
    # PASS 1A: Count Concept Occurrences Across the Entire Log
    # --------------------------------------------------------------------------
    print("\n[Pass 1A/2] Streaming Log_Problem.csv to select top concepts...")
    ucid_counts: Counter[str] = Counter()
    total_streamed_rows = 0

    for chunk in pd.read_csv(log_problem_csv, chunksize=chunksize, usecols=["ucid"]):
        total_streamed_rows += len(chunk)
        ucid_counts.update(chunk["ucid"].value_counts().to_dict())

    top_concepts = [c for c, _ in ucid_counts.most_common(top_n_concepts)]
    top_concepts_set = set(top_concepts)
    print(
        f"  Identified top {len(top_concepts)} concepts across {total_streamed_rows:,} total rows."
    )

    # --------------------------------------------------------------------------
    # PASS 1B: Count Student Interactions STRICTLY on Top Concepts
    # --------------------------------------------------------------------------
    print("\n[Pass 1B/2] Counting student interactions strictly on the top 150 concepts...")
    uuid_concept_counts: Counter[str] = Counter()

    for chunk in pd.read_csv(log_problem_csv, chunksize=chunksize, usecols=["uuid", "ucid"]):
        matched_mask = chunk["ucid"].isin(top_concepts_set)
        if matched_mask.any():
            uuid_concept_counts.update(chunk.loc[matched_mask, "uuid"].value_counts().to_dict())

    # Filter Students with >= min_interactions on top concepts
    eligible_students = sorted(
        [u for u, cnt in uuid_concept_counts.items() if cnt >= min_interactions]
    )
    print(
        f"  Eligible Students (>= {min_interactions} interactions on top concepts): "
        f"{len(eligible_students):,}"
    )

    # Deterministic Seeded Sampling
    rng = np.random.RandomState(seed)
    if len(eligible_students) > target_max_students:
        selected_students = sorted(
            rng.choice(eligible_students, size=target_max_students, replace=False).tolist()
        )
    else:
        selected_students = eligible_students

    selected_students_set = set(selected_students)
    print(f"  Selected Student Cohort: {len(selected_students):,} students")

    # --------------------------------------------------------------------------
    # PASS 2: Chunked Filtering & Monotonic Ordering
    # --------------------------------------------------------------------------
    print("\n[Pass 2/2] Streaming Log_Problem.csv to extract selected cohorts...")
    verified_cols = [
        "uuid",
        "ucid",
        "upid",
        "timestamp_TW",
        "is_correct",
        "total_sec_taken",
        "used_hint_cnt",
        "total_attempt_cnt",
    ]

    filtered_chunks: list[pd.DataFrame] = []
    global_row_offset = 0

    for chunk in pd.read_csv(log_problem_csv, chunksize=chunksize, usecols=verified_cols):
        chunk_len = len(chunk)
        chunk["orig_idx"] = np.arange(
            global_row_offset, global_row_offset + chunk_len, dtype=np.int64
        )
        global_row_offset += chunk_len

        # Filter strictly to target concepts and selected students
        mask = chunk["ucid"].isin(top_concepts_set) & chunk["uuid"].isin(selected_students_set)
        matched = chunk[mask]
        if not matched.empty:
            filtered_chunks.append(matched)

    if not filtered_chunks:
        raise ValueError("No matching records found during Pass 2 filtering!")

    df = pd.concat(filtered_chunks, ignore_index=True)
    n_extracted_raw = len(df)
    print(f"  Extracted {n_extracted_raw:,} interaction records for selected cohorts.")

    # --------------------------------------------------------------------------
    # DATA HYGIENE: Deduplication & Boundary Clipping
    # --------------------------------------------------------------------------
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=["uuid", "upid", "timestamp_TW", "is_correct"]).reset_index(
        drop=True
    )
    n_dropped_dups = n_before_dedup - len(df)
    print(f"  Data Hygiene: Dropped {n_dropped_dups:,} duplicate interaction rows.")

    # Verify and clip non-negative telemetry attributes
    df["total_sec_taken"] = np.maximum(0, df["total_sec_taken"].values)
    df["used_hint_cnt"] = np.maximum(0, df["used_hint_cnt"].values)
    df["total_attempt_cnt"] = np.maximum(0, df["total_attempt_cnt"].values)

    # Convert timestamps to exact Unix seconds (seconds resolution)
    df["timestamp_sec"] = (
        pd.to_datetime(df["timestamp_TW"], utc=True)
        .to_numpy(dtype="datetime64[s]")
        .astype(np.int64)
    )

    # Sort deterministically with monotonic orig_idx tie-breaker
    df = df.sort_values(by=["uuid", "timestamp_sec", "orig_idx"]).reset_index(drop=True)

    # Truncation to max_interactions per student sequence
    n_before_cap = len(df)
    if max_interactions > 0:
        df["interaction_rank"] = df.groupby("uuid").cumcount()
        df = df[df["interaction_rank"] < max_interactions].copy().reset_index(drop=True)
        df = df.drop(columns=["interaction_rank"])
    n_capped_tail_rows = n_before_cap - len(df)
    print(
        f"  Sequence Cap: Dropped {n_capped_tail_rows:,} tail rows "
        f"exceeding max_interactions={max_interactions}."
    )

    # Filter out students whose remaining interactions fell below min_interactions
    n_before_min = len(df)
    seq_counts = df.groupby("uuid").size()
    valid_students_after_cap = set(seq_counts[seq_counts >= min_interactions].index)
    df = df[df["uuid"].isin(valid_students_after_cap)].copy().reset_index(drop=True)
    n_dropped_short_students = n_before_min - len(df)
    if n_dropped_short_students > 0:
        print(
            f"  Short Student Filter: Dropped {n_dropped_short_students:,} rows "
            f"from students below min_interactions."
        )

    # Report sequence length statistics
    seq_lens = df.groupby("uuid").size()
    seq_stats = {
        "min": int(seq_lens.min()),
        "median": float(seq_lens.median()),
        "p95": float(np.percentile(seq_lens, 95)),
        "max": int(seq_lens.max()),
    }
    print(
        f"  Sequence lengths per student: min={seq_stats['min']}, "
        f"median={seq_stats['median']:.1f}, p95={seq_stats['p95']:.1f}, max={seq_stats['max']}"
    )

    # --------------------------------------------------------------------------
    # Join Curriculum Hierarchy & Difficulty from Info_Content.csv
    # --------------------------------------------------------------------------
    print("\nJoining curriculum taxonomy and concept difficulties from Info_Content.csv...")
    content_cols = ["ucid", "level1_id", "level2_id", "level3_id", "level4_id", "difficulty"]
    info_content = pd.read_csv(info_content_csv, usecols=content_cols)

    difficulty_map = {"unset": 0, "easy": 1, "normal": 2, "hard": 3}
    info_content["difficulty_id"] = (
        info_content["difficulty"]
        .fillna("unset")
        .astype(str)
        .str.lower()
        .map(lambda d: difficulty_map.get(d, 0))
        .astype(np.int64)
    )

    df = df.merge(info_content, on="ucid", how="left")
    df["difficulty_id"] = df["difficulty_id"].fillna(0).astype(np.int64)

    # --------------------------------------------------------------------------
    # Feature Engineering (Outlier Clipping & Log Scaling)
    # --------------------------------------------------------------------------
    print("Computing log-scaled interaction dynamics (elapsed time, attempts, dwell, hints)...")
    df["is_correct"] = df["is_correct"].astype(np.int64)
    df["is_hint_used"] = (df["used_hint_cnt"] > 0).astype(np.int64)
    df["log_hint_cnt"] = np.log1p(np.clip(df["used_hint_cnt"].values, 0, 50)).astype(np.float32)
    df["log_sec"] = np.log1p(np.clip(df["total_sec_taken"].values, 0, 3600)).astype(np.float32)
    df["log_attempt"] = np.log1p(np.clip(df["total_attempt_cnt"].values, 0, 50)).astype(
        np.float32
    )

    # Elapsed time between successive interactions: delta_t = log1p(max(0, tau_t - tau_{t-1}))
    prev_student_time = df.groupby("uuid")["timestamp_sec"].diff().fillna(0.0).values
    df["delta_t"] = np.log1p(np.maximum(0.0, prev_student_time)).astype(np.float32)

    # --------------------------------------------------------------------------
    # Student Split (Student-Disjoint 70/15/15)
    # --------------------------------------------------------------------------
    test_ratio = 1.0 - train_ratio - val_ratio
    print(
        f"\nPartitioning students ({train_ratio:.0%} Train / {val_ratio:.0%} Val / "
        f"{test_ratio:.0%} Test)..."
    )
    unique_students = np.sort(df["uuid"].unique())
    split_rng = np.random.RandomState(seed)
    split_rng.shuffle(unique_students)

    n_students = len(unique_students)
    n_train = int(n_students * train_ratio)
    n_val = int(n_students * val_ratio)

    train_students = set(unique_students[:n_train])
    val_students = set(unique_students[n_train : n_train + n_val])
    test_students = set(unique_students[n_train + n_val :])

    train_df = df[df["uuid"].isin(train_students)].copy().reset_index(drop=True)
    val_df = df[df["uuid"].isin(val_students)].copy().reset_index(drop=True)
    test_df = df[df["uuid"].isin(test_students)].copy().reset_index(drop=True)

    n_final_split_total = len(train_df) + len(val_df) + len(test_df)
    assert (
        n_extracted_raw - n_dropped_dups - n_capped_tail_rows - n_dropped_short_students
        == n_final_split_total
    ), (
        f"Row reconciliation arithmetic mismatch! "
        f"{n_extracted_raw} - {n_dropped_dups} - {n_capped_tail_rows} - {n_dropped_short_students} "
        f"!= {n_final_split_total}"
    )

    print("\n" + "=" * 78)
    print("EXACT INTERACTION ROW RECONCILIATION")
    print("=" * 78)
    print(f"  Extracted Raw Rows       : {n_extracted_raw:>10,}")
    print(f"  - Duplicate Rows Dropped : {n_dropped_dups:>10,}")
    print(f"  - Capped Tail Rows (1000): {n_capped_tail_rows:>10,}")
    print(f"  - Dropped Short Students : {n_dropped_short_students:>10,}")
    print(f"  = Final Split Total      : {n_final_split_total:>10,}")
    print(f"  Train: {len(train_students):,} students, {len(train_df):,} interactions")
    print(f"  Val  : {len(val_students):,} students, {len(val_df):,} interactions")
    print(f"  Test : {len(test_students):,} students, {len(test_df):,} interactions")
    print("=" * 78)

    # --------------------------------------------------------------------------
    # Zero-Leakage Vocabulary & Scaler Fitting (STRICTLY on Train Split)
    # --------------------------------------------------------------------------
    print("\nFitting concept vocab, problem indexer, and StandardScaler STRICTLY on Train...")

    # Concept Index: Map all tracked top_concepts deterministically to 0..N-1
    concept2idx = {c: i for i, c in enumerate(top_concepts)}
    num_concepts = len(concept2idx)

    # Problem Index: Reserve index 0 for unseen <UNK> test problems
    train_problems = sorted(train_df["upid"].unique().tolist())
    problem2idx = {"<UNK>": 0}
    for p in train_problems:
        problem2idx[p] = len(problem2idx)
    num_problems = len(problem2idx)

    # Map vocabularies across splits with strict range verification
    for s_name, split_data in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_data["concept_idx"] = split_data["ucid"].map(concept2idx).astype(np.int64)
        assert not split_data["concept_idx"].isna().any(), f"NaN concept_idx in {s_name}"
        assert (split_data["concept_idx"] >= 0).all(), f"Negative concept_idx in {s_name}"
        assert (split_data["concept_idx"] < num_concepts).all(), (
            f"Out-of-bounds concept_idx in {s_name}"
        )

        split_data["problem_idx"] = (
            split_data["upid"].map(lambda p: problem2idx.get(p, 0)).astype(np.int64)
        )
        assert not split_data["problem_idx"].isna().any(), f"NaN problem_idx in {s_name}"
        assert (split_data["problem_idx"] >= 0).all(), f"Negative problem_idx in {s_name}"
        assert (split_data["problem_idx"] < num_problems).all(), (
            f"Out-of-bounds problem_idx in {s_name}"
        )

    print(f"  Concept Vocab: {num_concepts} tracked concepts (0 to {num_concepts - 1})")
    print(f"  Problem Vocab: {num_problems:,} problems including <UNK>=0 (0 to {num_problems - 1})")

    # Continuous Feature Normalization
    continuous_cols = ["log_sec", "log_hint_cnt", "log_attempt", "delta_t"]
    scaler = StandardScaler()
    scaler.fit(train_df[continuous_cols])

    norm_cols = [f"norm_{col}" for col in continuous_cols]
    train_df[norm_cols] = scaler.transform(train_df[continuous_cols]).astype(np.float32)
    val_df[norm_cols] = scaler.transform(val_df[continuous_cols]).astype(np.float32)
    test_df[norm_cols] = scaler.transform(test_df[continuous_cols]).astype(np.float32)

    # --------------------------------------------------------------------------
    # Construct Delayed-Retention Prediction Targets
    # --------------------------------------------------------------------------
    print("\nConstructing delayed-retention prediction targets across splits...")
    train_df, train_retention = compute_delayed_retention(train_df)
    val_df, val_retention = compute_delayed_retention(val_df)
    test_df, test_retention = compute_delayed_retention(test_df)

    # --------------------------------------------------------------------------
    # Hygiene Verification: Assert Zero NaNs / Infs Across All DataFrames
    # --------------------------------------------------------------------------
    datasets_to_verify = [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
        ("train_retention", train_retention),
        ("val_retention", val_retention),
        ("test_retention", test_retention),
    ]
    for d_name, d_obj in datasets_to_verify:
        num_cols = d_obj.select_dtypes(include=[np.number]).columns
        nan_sum = int(d_obj[num_cols].isna().sum().sum())
        inf_sum = int(np.isinf(d_obj[num_cols].to_numpy()).sum())
        if nan_sum > 0 or inf_sum > 0:
            raise ValueError(
                f"Data hygiene assertion failed for {d_name}: {nan_sum} NaNs, {inf_sum} Infs found!"
            )

    print("  Data Hygiene: Verified 0 NaNs and 0 Infs across all numeric columns.")

    # --------------------------------------------------------------------------
    # Save Parquet Datasets & Metadata
    # --------------------------------------------------------------------------
    print(f"\nPersisting processed artifacts to {data_proc_dir}...")
    train_df.to_parquet(data_proc_dir / "train.parquet", index=False)
    val_df.to_parquet(data_proc_dir / "val.parquet", index=False)
    test_df.to_parquet(data_proc_dir / "test.parquet", index=False)

    train_retention.to_parquet(data_proc_dir / "train_retention.parquet", index=False)
    val_retention.to_parquet(data_proc_dir / "val_retention.parquet", index=False)
    test_retention.to_parquet(data_proc_dir / "test_retention.parquet", index=False)

    metadata = {
        "dataset_name": "Junyi Academy 2020",
        "seed": seed,
        "fast_dev_run": fast_dev_run,
        "num_concepts": len(concept2idx),
        "num_problems": len(problem2idx),
        "concept2idx": concept2idx,
        "difficulty_map": difficulty_map,
        "retention_buckets": {
            0: "<1h",
            1: "1h-24h",
            2: "1d-7d",
            3: ">7d",
        },
        "scaler": {
            "feature_names": continuous_cols,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "var": scaler.var_.tolist(),
        },
        "sequence_length_stats": seq_stats,
        "row_reconciliation": {
            "n_extracted_raw": n_extracted_raw,
            "n_dropped_dups": n_dropped_dups,
            "n_capped_tail_rows": n_capped_tail_rows,
            "n_dropped_short_students": n_dropped_short_students,
            "n_final_split_total": n_final_split_total,
            "formula": (
                "extracted_rows - duplicate_rows - capped_tail_rows - "
                "dropped_short_students == final_split_total"
            ),
            "verified": True,
        },
        "data_hygiene": {
            "duplicate_rows_dropped": n_dropped_dups,
            "zero_nans_verified": True,
            "zero_infs_verified": True,
        },
        "cohort_counts": {
            "train_students": len(train_students),
            "val_students": len(val_students),
            "test_students": len(test_students),
            "train_interactions": len(train_df),
            "val_interactions": len(val_df),
            "test_interactions": len(test_df),
            "train_retention_pairs": len(train_retention),
            "val_retention_pairs": len(val_retention),
            "test_retention_pairs": len(test_retention),
        },
        "class_balance": {
            "train_correct_pct": float(train_df["is_correct"].mean()),
            "val_correct_pct": float(val_df["is_correct"].mean()),
            "test_correct_pct": float(test_df["is_correct"].mean()),
        },
    }

    with open(data_proc_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(data_proc_dir / "vocab_concept.json", "w", encoding="utf-8") as f:
        json.dump(concept2idx, f, indent=2)

    with open(data_proc_dir / "vocab_problem.json", "w", encoding="utf-8") as f:
        json.dump(problem2idx, f, indent=2)

    # --------------------------------------------------------------------------
    # Summary Report
    # --------------------------------------------------------------------------
    train_acc = train_df["is_correct"].mean()
    val_acc = val_df["is_correct"].mean()
    test_acc = test_df["is_correct"].mean()
    total_ret = len(train_retention) + len(val_retention) + len(test_retention)

    print("\n" + "=" * 78)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 78)
    print(f"Concepts Tracked (N) : {len(concept2idx)}")
    print(f"Problem Vocab Size   : {len(problem2idx):,} (including <UNK>=0)")
    print(
        f"Students Cohort Size : {len(unique_students):,} "
        f"(Train: {len(train_students):,}, Val: {len(val_students):,}, "
        f"Test: {len(test_students):,})"
    )
    print(
        f"Total Interactions   : {len(df):,} "
        f"(Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,})"
    )
    print(f"Class Balance (Acc)  : Train={train_acc:.3f}, Val={val_acc:.3f}, Test={test_acc:.3f}")
    print(f"Retention Pairs Total: {total_ret:,}")

    print("\n" + "-" * 78)
    print("RETENTION DELAY BUCKET BREAKDOWN (ACROSS TRAIN, VAL, TEST)")
    print("-" * 78)
    bucket_order = ["<1h", "1h-24h", "1d-7d", ">7d"]
    splits_ret = [
        ("Train", train_retention),
        ("Val", val_retention),
        ("Test", test_retention),
    ]
    for s_name, s_ret in splits_ret:
        b_counts = s_ret["retention_bucket_name"].value_counts().to_dict()
        n_total_s = len(s_ret)
        b_summary = " | ".join(
            [
                f"{b}: {b_counts.get(b, 0):>5,} "
                f"({b_counts.get(b, 0) / max(1, n_total_s) * 100:>5.1f}%)"
                for b in bucket_order
            ]
        )
        print(f"  {s_name:<5} (Total {n_total_s:>5,}): {b_summary}")
    print("=" * 78)

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TPA-GKT Preprocessing Pipeline")
    parser.add_argument(
        "--config", type=str, default="services/gnn_kt/config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--fast-dev-run", action="store_true", help="Run with small cohort for fast testing"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    args = parser.parse_args()

    run_preprocessing(
        config_path=args.config,
        fast_dev_run=args.fast_dev_run,
        override_seed=args.seed,
    )
