#!/usr/bin/env python3
"""
inspect_data.py - Schema Inspection & Feature Availability for TPA-GKT

Inspects Junyi Academy 2020 raw CSV datasets:
- Info_Content.csv (loaded in full, ~389 KB)
- Info_UserData.csv (first 1,000 rows, ~6 MB file)
- Log_Problem.csv (strictly nrows=1000, ~2.95 GB file - NEVER loaded fully into RAM)

Outputs:
- Formatted console schema report.
- Markdown table saved to services/gnn_kt/outputs/feature_availability.md
  for Section IV.C of Paper 1.
"""

from pathlib import Path
from typing import Any
import pandas as pd
import yaml


def load_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def format_samples(series: pd.Series, max_samples: int = 3) -> str:
    non_null = series.dropna().unique()
    samples = [str(x) for x in non_null[:max_samples]]
    if not samples:
        return "None"
    joined = ", ".join(samples)
    return f"`{joined}`" if len(joined) < 45 else f"`{joined[:42]}...`"


def inspect_dataframe(df: pd.DataFrame, name: str, source_file: str) -> list[dict[str, Any]]:
    print(f"\n{'='*78}")
    print(f" Dataset: {name} (from {source_file})")
    print(f" Sample Rows: {len(df):,} | Total Columns: {len(df.columns)}")
    print(f"{'='*78}")

    rows_info = []
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        null_count = int(s.isnull().sum())
        null_pct = (null_count / len(s)) * 100.0 if len(s) > 0 else 0.0
        samples_str = format_samples(s)

        print(f" - {col:<26} | {dtype:<10} | nulls: {null_count:>4} ({null_pct:>5.1f}%) | samples: {samples_str}")

        rows_info.append({
            "dataset": name,
            "source_file": source_file,
            "column": col,
            "dtype": dtype,
            "null_count": null_count,
            "null_pct": null_pct,
            "samples": samples_str,
        })
    return rows_info


def find_column(candidates: list[str], available_cols: list[str]) -> str | None:
    available_lower = {col.lower(): col for col in available_cols}
    for cand in candidates:
        if cand.lower() in available_lower:
            return available_lower[cand.lower()]
    # Substring search fallback
    for cand in candidates:
        for a_low, orig in available_lower.items():
            if cand.lower() in a_low:
                return orig
    return None


def main() -> None:
    config = load_config()
    base_dir = Path(__file__).resolve().parent.parent
    raw_rel = config.get("data_raw_dir", "../../data/raw")
    raw_dir = (base_dir / raw_rel).resolve()
    outputs_dir = (base_dir / config.get("outputs_dir", "./outputs")).resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[inspect_data] Base dir: {base_dir}")
    print(f"[inspect_data] Raw data dir: {raw_dir}")
    print(f"[inspect_data] Outputs dir: {outputs_dir}")

    content_path = raw_dir / "Info_Content.csv"
    user_path = raw_dir / "Info_UserData.csv"
    log_path = raw_dir / "Log_Problem.csv"

    # 1. Inspect Info_Content.csv (read in full)
    content_info: list[dict[str, Any]] = []
    if content_path.exists():
        df_content = pd.read_csv(content_path)
        content_info = inspect_dataframe(df_content, "Info_Content", "Info_Content.csv")
    else:
        print(f"[WARN] File not found: {content_path}")
        df_content = pd.DataFrame()

    # 2. Inspect Info_UserData.csv (first 1,000 rows)
    user_info: list[dict[str, Any]] = []
    if user_path.exists():
        df_user = pd.read_csv(user_path, nrows=1000)
        user_info = inspect_dataframe(df_user, "Info_UserData", "Info_UserData.csv")
    else:
        print(f"[WARN] File not found: {user_path}")
        df_user = pd.DataFrame()

    # 3. Inspect Log_Problem.csv (STRICTLY nrows=1000, large file protection)
    log_info: list[dict[str, Any]] = []
    if log_path.exists():
        df_log = pd.read_csv(log_path, nrows=1000)
        log_info = inspect_dataframe(df_log, "Log_Problem", "Log_Problem.csv")
    else:
        print(f"[WARN] File not found: {log_path}")
        df_log = pd.DataFrame()

    # 4. Map required TPA-GKT features
    # Required features defined in SPEC.md
    feature_requirements = [
        {
            "feature": "Student ID (`uuid`)",
            "role": "Learner identifier for student-disjoint split & sequence grouping",
            "candidates": ["uuid", "user_id", "student_id"],
            "expected_in": "Log_Problem.csv",
            "df": df_log,
        },
        {
            "feature": "Concept ID (`ucid`)",
            "role": "Knowledge Component / Concept node in dual graph",
            "candidates": ["ucid", "concept_id", "kc_id", "content_id"],
            "expected_in": "Log_Problem.csv & Info_Content.csv",
            "df": df_log,
        },
        {
            "feature": "Problem ID (`upid`)",
            "role": "Exercise / item instance identifier",
            "candidates": ["upid", "problem_id", "item_id", "exercise_id"],
            "expected_in": "Log_Problem.csv",
            "df": df_log,
        },
        {
            "feature": "Timestamp (`timestamp_TW`)",
            "role": "Interaction time for temporal ordering & elapsed time delta (Δt)",
            "candidates": ["timestamp_TW", "timestamp", "time", "created_at", "problem_time"],
            "expected_in": "Log_Problem.csv",
            "df": df_log,
        },
        {
            "feature": "Correctness (`is_correct`)",
            "role": "Binary ground-truth attempt correctness label for Head 1",
            "candidates": ["is_correct", "correct", "is_correct_first_attempt", "score"],
            "expected_in": "Log_Problem.csv",
            "df": df_log,
        },
        {
            "feature": "Response Time (`total_sec_taken`)",
            "role": "Continuous dwell / problem latency feature",
            "candidates": ["total_sec_taken", "response_time", "total_time", "duration_sec", "time_taken"],
            "expected_in": "Log_Problem.csv",
            "df": df_log,
        },
        {
            "feature": "Hint Usage (`total_hint_cnt`)",
            "role": "Scaffolding indicator feature",
            "candidates": ["total_hint_cnt", "hint_count", "used_hint", "hint", "hints_requested"],
            "expected_in": "Log_Problem.csv",
            "df": df_log,
        },
        {
            "feature": "Attempt Count (`total_attempt_cnt`)",
            "role": "Trial frequency prior to problem resolution",
            "candidates": ["total_attempt_cnt", "attempt_count", "attempts", "try_count"],
            "expected_in": "Log_Problem.csv",
            "df": df_log,
        },
        {
            "feature": "Curriculum Level 1 (`level1_id`)",
            "role": "Top-level subject domain in hierarchical graph A_hier",
            "candidates": ["level1_id", "subject_id", "domain_id", "level1"],
            "expected_in": "Info_Content.csv",
            "df": df_content,
        },
        {
            "feature": "Curriculum Level 2 (`level2_id`)",
            "role": "Intermediate topic structure in hierarchical graph A_hier",
            "candidates": ["level2_id", "topic_id", "subtopic_id", "level2"],
            "expected_in": "Info_Content.csv",
            "df": df_content,
        },
        {
            "feature": "Concept Difficulty (`difficulty`)",
            "role": "Intrinsic item / concept difficulty prior",
            "candidates": ["difficulty", "content_difficulty", "level", "learning_stage"],
            "expected_in": "Info_Content.csv",
            "df": df_content,
        },
    ]

    mapped_rows = []
    print(f"\n{'='*78}")
    print(" TPA-GKT Feature Mapping Analysis")
    print(f"{'='*78}")

    for req in feature_requirements:
        df_target = req["df"]
        matched_col = find_column(req["candidates"], list(df_target.columns)) if not df_target.empty else None

        if matched_col:
            s = df_target[matched_col]
            dtype = str(s.dtype)
            null_pct = (s.isnull().sum() / len(s)) * 100.0 if len(s) > 0 else 0.0
            samples_str = format_samples(s)
            status = "VERIFIED"
            col_display = f"`{matched_col}`"
        else:
            dtype = "-"
            null_pct = 0.0
            samples_str = "-"
            status = "MISSING"
            col_display = "**MISSING**"

        print(f" {req['feature']:<32} -> {col_display:<22} [{status}] ({req['expected_in']})")

        mapped_rows.append({
            "feature": req["feature"],
            "role": req["role"],
            "expected_file": req["expected_in"],
            "verified_col": col_display,
            "dtype": dtype,
            "null_pct": f"{null_pct:.1f}%" if matched_col else "-",
            "samples": samples_str,
            "status": status,
        })

    # 5. Build Markdown table for feature_availability.md
    md_lines = [
        "# TPA-GKT Feature Availability & Schema Mapping Report",
        "",
        "> **Dataset**: Junyi Academy 2020 (`data/raw/`)  ",
        "> **Target**: Section IV.C of Paper 1 (*Empirical Telemetry & Feature Availability*)  ",
        f"> **Generated By**: `inspect_data.py` (Memory-safe streaming with `nrows=1000` on 2.95 GB Log_Problem.csv)",
        "",
        "## 1. Summary of Raw Datasets Inspected",
        "",
        "| Dataset File | File Size | Rows Sampled / Total | Verified Columns Count | Description |",
        "|---|---|---|---|---|",
        f"| `Info_Content.csv` | ~389 KB | {len(df_content):,} (full) | {len(df_content.columns)} | Knowledge component taxonomy & curriculum hierarchy |",
        f"| `Info_UserData.csv` | ~6.36 MB | {len(df_user):,} (sample) | {len(df_user.columns)} | Student demographic and platform profile telemetry |",
        f"| `Log_Problem.csv` | ~2.95 GB | {len(df_log):,} (sample) | {len(df_log.columns)} | Fine-grained sequential problem interaction logs |",
        "",
        "## 2. Core TPA-GKT Feature Mapping (Paper Table 1)",
        "",
        "| Required TPA-GKT Feature | Role in Architecture | Expected Source | Verified Column Name | Inferred Dtype | Missing % | Sample Values | Verification Status |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for m in mapped_rows:
        status_badge = "✅ VERIFIED" if m["status"] == "VERIFIED" else "❌ MISSING"
        md_lines.append(
            f"| {m['feature']} | {m['role']} | `{m['expected_file']}` | {m['verified_col']} | `{m['dtype']}` | {m['null_pct']} | {m['samples']} | {status_badge} |"
        )

    md_lines.extend([
        "",
        "## 3. Detailed Column Inventory Across Datasets",
        "",
        "### A. `Info_Content.csv` (Curriculum Taxonomy)",
        "",
        "| Column | Inferred Dtype | Null Count | Null % | Sample Values |",
        "|---|---|---|---|---|",
    ])
    for c in content_info:
        md_lines.append(f"| `{c['column']}` | `{c['dtype']}` | {c['null_count']} | {c['null_pct']:.1f}% | {c['samples']} |")

    md_lines.extend([
        "",
        "### B. `Info_UserData.csv` (Student Attributes)",
        "",
        "| Column | Inferred Dtype | Null Count | Null % | Sample Values |",
        "|---|---|---|---|---|",
    ])
    for c in user_info:
        md_lines.append(f"| `{c['column']}` | `{c['dtype']}` | {c['null_count']} | {c['null_pct']:.1f}% | {c['samples']} |")

    md_lines.extend([
        "",
        "### C. `Log_Problem.csv` (Interaction Logs, First 1,000 Rows Sampled)",
        "",
        "| Column | Inferred Dtype | Null Count | Null % | Sample Values |",
        "|---|---|---|---|---|",
    ])
    for c in log_info:
        md_lines.append(f"| `{c['column']}` | `{c['dtype']}` | {c['null_count']} | {c['null_pct']:.1f}% | {c['samples']} |")

    md_lines.extend([
        "",
        "## 4. Key Architectural Implications for `preprocess.py`",
        "",
        "1. **Primary Join Key**: Both `Log_Problem.csv` and `Info_Content.csv` share `ucid` (Unique Concept ID), which directly maps interaction logs into knowledge component nodes.",
        "2. **Temporal Alignment**: `timestamp_TW` (or equivalent time column) provides exact microsecond/second timestamps to compute continuous time delta $\\Delta t = t_k - t_{k-1}$ for forgetting decay modeling.",
        "3. **Prerequisite Graph (`A_hier`)**: `level1_id` and `level2_id` in `Info_Content.csv` provide deterministic curriculum subtrees to establish prerequisite parent/child and sibling edges.",
        "4. **Transition Graph (`A_trans`)**: Computed purely from sequential transitions in `Log_Problem.csv` across the Train split.",
    ])

    out_file = outputs_dir / "feature_availability.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\n[inspect_data] Saved feature availability report to: {out_file}")


if __name__ == "__main__":
    main()
