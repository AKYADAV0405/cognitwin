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
    chunksize = int(cfg.get("chunksize", 500000))
    train_ratio = float(cfg.get("train_ratio", 0.70))
    val_ratio = float(cfg.get("val_ratio", 0.15))

    target_max_students = (
        int(cfg["fast_dev_max_students"]) if fast_dev_run else int(cfg["max_students"])
    )

    print("=" * 78)
    print("TPA-GKT PREPROCESSING: ZERO-LEAKAGE CHUNKED STREAMING")
    print(f"Data Raw Dir     : {data_raw_dir}")
    print(f"Data Processed   : {data_proc_dir}")
    print(f"Fast Dev Run     : {fast_dev_run} (Max Students: {target_max_students:,})")
    print(f"Top Concepts (N) : {top_n_concepts}")
    print(f"Min Interactions : {min_interactions}")
    print(f"Random Seed      : {seed}")
    print("=" * 78)

    # --------------------------------------------------------------------------
    # PASS 1: Chunked Counting (Memory-Safe Concept & Student Selection)
    # --------------------------------------------------------------------------
    print("\n[Pass 1/2] Streaming Log_Problem.csv to count concept and student occurrences...")
    ucid_counts: Counter[str] = Counter()
    uuid_counts: Counter[str] = Counter()

    total_streamed_rows = 0
    chunk_idx = 0
    for chunk in pd.read_csv(log_problem_csv, chunksize=chunksize, usecols=["uuid", "ucid"]):
        total_streamed_rows += len(chunk)
        chunk_idx += 1
        ucid_counts.update(chunk["ucid"].value_counts().to_dict())
        uuid_counts.update(chunk["uuid"].value_counts().to_dict())
        if chunk_idx % 10 == 0:
            print(f"  Processed {total_streamed_rows:,} rows across {chunk_idx} chunks...")

    print(
        f"  Total Rows: {total_streamed_rows:,} across {len(ucid_counts):,} concepts "
        f"and {len(uuid_counts):,} students."
    )

    # Select Top N Concepts
    top_concepts = [c for c, _ in ucid_counts.most_common(top_n_concepts)]
    top_concepts_set = set(top_concepts)

    # Filter Students with >= min_interactions
    eligible_students = sorted(
        [u for u, cnt in uuid_counts.items() if cnt >= min_interactions]
    )
    print(f"  Eligible Students (>={min_interactions} total): {len(eligible_students):,}")

    # Deterministic Seeded Subsampling
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
    print("\n[Pass 2/2] Streaming Log_Problem.csv to filter cohorts & preserve tie-breakers...")
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

        # Filter to target concepts and selected students
        mask = chunk["ucid"].isin(top_concepts_set) & chunk["uuid"].isin(selected_students_set)
        matched = chunk[mask]
        if not matched.empty:
            filtered_chunks.append(matched)

    if not filtered_chunks:
        raise ValueError("No matching records found during Pass 2 filtering!")

    df = pd.concat(filtered_chunks, ignore_index=True)
    print(f"  Extracted {len(df):,} interaction records for selected cohorts.")

    # Convert timestamps to UTC Unix seconds (tie-breaker: uuid, timestamp_sec, orig_idx)
    df["timestamp_sec"] = pd.to_datetime(df["timestamp_TW"], utc=True).astype("int64") // 10**9
    df = df.sort_values(by=["uuid", "timestamp_sec", "orig_idx"]).reset_index(drop=True)

    # Enforce min_interactions constraint on target concepts
    student_concept_counts = df["uuid"].value_counts()
    valid_students = set(
        student_concept_counts[student_concept_counts >= min_interactions].index.tolist()
    )
    df = df[df["uuid"].isin(valid_students)].copy().reset_index(drop=True)
    print(
        f"  Cohort with >={min_interactions} on top concepts: "
        f"{len(valid_students):,} students ({len(df):,} interactions)."
    )

    # Optional Truncation to max_interactions per student sequence
    if max_interactions > 0:
        df["interaction_rank"] = df.groupby("uuid").cumcount()
        df = df[df["interaction_rank"] < max_interactions].copy().reset_index(drop=True)
        df = df.drop(columns=["interaction_rank"])

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

    print(f"  Train: {len(train_students):,} students, {len(train_df):,} interactions")
    print(f"  Val  : {len(val_students):,} students, {len(val_df):,} interactions")
    print(f"  Test : {len(test_students):,} students, {len(test_df):,} interactions")

    # --------------------------------------------------------------------------
    # Zero-Leakage Vocabulary & Scaler Fitting (STRICTLY on Train Split)
    # --------------------------------------------------------------------------
    print("\nFitting concept vocab, problem indexer, and StandardScaler STRICTLY on Train...")

    # Concept Index: Map all concepts present in Train deterministically to 0..N-1
    train_concepts = sorted(train_df["ucid"].unique().tolist())
    concept2idx = {c: i for i, c in enumerate(train_concepts)}

    # Problem Index: Reserve index 0 for unseen <UNK> test problems
    train_problems = sorted(train_df["upid"].unique().tolist())
    problem2idx = {"<UNK>": 0}
    for p in train_problems:
        problem2idx[p] = len(problem2idx)

    # Map vocabularies across splits
    for split_data in [train_df, val_df, test_df]:
        split_data["concept_idx"] = (
            split_data["ucid"].map(lambda c: concept2idx.get(c, -1)).astype(np.int64)
        )
        split_data["problem_idx"] = (
            split_data["upid"].map(lambda p: problem2idx.get(p, 0)).astype(np.int64)
        )

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
    print("Retention Target Distribution (Train):")
    print(train_retention["retention_bucket_name"].value_counts().to_string())
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
