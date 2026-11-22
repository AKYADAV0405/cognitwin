"""
Phase 2: Retention Target & Temporal Leakage Forensic Audit Module.

Conducts research-grade forensic verification of delayed retention pairs on Junyi Academy:
  1. Validates source and target interaction IDs, timestamps, and concept matching.
  2. Verifies 100% zero-leakage same-concept isolation between source and target.
  3. Disentangles intervening practice into:
       - Isolated Retention (0 intervening practice)
       - Related-Concept Interference (intervening practice on graph neighbors)
       - Unrelated-Concept Interference (intervening practice on unconnected concepts)
  4. Analyzes timestamp quantization (15-minute grid) and fine-grained <1h sub-intervals.
  5. Computes sample counts, empirical correctness rates, and Retention AUC across all strata.
  6. Saves all outputs cleanly to outputs/validation_fixed/retention_audit/ without overwriting.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parents[3])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from services.gnn_kt.src.dataset import create_dataloader  # noqa: E402
from services.gnn_kt.src.model import TPAGKT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Forensic Pair Extraction & Zero-Leakage Verification
# ==============================================================================
def classify_delay_interval(delta_sec: float) -> tuple[str, int]:
    """
    Classify elapsed time delta_sec into fine-grained delay interval and legacy bucket ID.
    Intervals:
      - '<15m'   : delta_sec <= 900 (quantized immediate within 15 min)
      - '15m-60m': 900 < delta_sec <= 3600 (within-session review)
      - '1h-24h' : 3600 < delta_sec <= 86400 (same-day review)
      - '1d-7d'  : 86400 < delta_sec <= 604800 (multi-day spaced review)
      - '>7d'    : delta_sec > 604800 (long-term retention)
    """
    if delta_sec <= 900:
        return "<15m", 0
    elif delta_sec <= 3600:
        return "15m-60m", 0
    elif delta_sec <= 86400:
        return "1h-24h", 1
    elif delta_sec <= 604800:
        return "1d-7d", 2
    else:
        return ">7d", 3


def extract_and_verify_retention_pairs(
    test_parquet: Path,
    graph_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Extract, verify, and stratify delayed-retention pairs directly from test.parquet.

    Returns:
      pairs_df: DataFrame containing every verified source-target retention pair.
      forensics: Summary statistics and zero-leakage compliance audit.
    """
    logger.info("Loading test interactions from %s...", test_parquet)
    df = pd.read_parquet(test_parquet)

    # Ensure deterministic chronological ordering
    df = df.sort_values(by=["uuid", "timestamp_sec", "orig_idx"]).reset_index(drop=True)
    df["row_idx"] = df.index

    # Load graph adjacency if available
    adj: dict[int, set[int]] = defaultdict(set)
    if graph_path is not None and graph_path.exists():
        logger.info("Loading concept graph topology from %s...", graph_path)
        loaded = torch.load(graph_path, weights_only=False)
        if hasattr(loaded, "edge_index"):
            e_idx = loaded.edge_index.cpu().numpy()
            for u, v in zip(e_idx[0], e_idx[1], strict=False):
                adj[int(u)].add(int(v))
                adj[int(v)].add(int(u))
        elif isinstance(loaded, torch.Tensor):
            if loaded.dim() == 2 and loaded.shape[0] == 2:
                e_idx = loaded.cpu().numpy()
                for u, v in zip(e_idx[0], e_idx[1], strict=False):
                    adj[int(u)].add(int(v))
                    adj[int(v)].add(int(u))
            elif loaded.dim() == 2 and loaded.shape[0] == loaded.shape[1]:
                nz = torch.nonzero(loaded > 0, as_tuple=False).cpu().numpy()
                for u, v in nz:
                    adj[int(u)].add(int(v))
                    adj[int(v)].add(int(u))

    pairs: list[dict[str, Any]] = []
    same_concept_leakage_count = 0
    negative_delta_count = 0

    # Group interactions by (uuid, ucid) to find consecutive attempts on the same concept
    grouped = df.groupby(["uuid", "ucid"], sort=False)

    for (uuid_val, ucid_val), group in grouped:
        if len(group) < 2:
            continue

        rows = group["row_idx"].values
        for k in range(len(rows) - 1):
            src_row = rows[k]
            tgt_row = rows[k + 1]

            src_orig = int(df.loc[src_row, "orig_idx"])
            tgt_orig = int(df.loc[tgt_row, "orig_idx"])
            src_time = float(df.loc[src_row, "timestamp_sec"])
            tgt_time = float(df.loc[tgt_row, "timestamp_sec"])
            delta_sec = tgt_time - src_time

            # Verification rule 1: Delta t must be strictly positive
            if delta_sec <= 0:
                negative_delta_count += 1
                continue

            src_c = int(df.loc[src_row, "concept_idx"])
            tgt_c = int(df.loc[tgt_row, "concept_idx"])
            assert src_c == tgt_c, f"Concept mismatch: {src_c} != {tgt_c}"

            src_p = (
                int(df.loc[src_row, "problem_idx"])
                if "problem_idx" in df.columns
                else int(df.loc[src_row, "upid"])
            )
            tgt_p = (
                int(df.loc[tgt_row, "problem_idx"])
                if "problem_idx" in df.columns
                else int(df.loc[tgt_row, "upid"])
            )
            tgt_corr = int(df.loc[tgt_row, "is_correct"])

            # Intervening practice analysis between src_row and tgt_row
            n_intervening = tgt_row - src_row - 1
            intervening_category = "isolated"
            intervening_concept_count = 0

            if n_intervening > 0:
                intervening_slice = df.loc[src_row + 1 : tgt_row - 1]
                inter_concepts = set(intervening_slice["concept_idx"].values)
                intervening_concept_count = len(inter_concepts)

                # Verification rule 2: Target must be the immediately next attempt on concept c
                if src_c in inter_concepts:
                    same_concept_leakage_count += 1
                    continue

                # Check graph connectivity
                neighbors = adj.get(src_c, set())
                has_related = any(c in neighbors for c in inter_concepts)
                if has_related:
                    intervening_category = "related_interference"
                else:
                    intervening_category = "unrelated_interference"

            # Fine-grained delay bucket assignment
            interval_label, legacy_bucket = classify_delay_interval(delta_sec)

            pairs.append(
                {
                    "uuid": str(uuid_val),
                    "ucid": str(ucid_val),
                    "concept_idx": src_c,
                    "source_concept": src_c,
                    "target_concept": src_c,
                    "src_orig_idx": src_orig,
                    "tgt_orig_idx": tgt_orig,
                    "src_timestamp": src_time,
                    "tgt_timestamp": tgt_time,
                    "source_timestamp": src_time,
                    "target_timestamp": tgt_time,
                    "delta_sec": delta_sec,
                    "target_correctness": tgt_corr,
                    "target_correct": tgt_corr,
                    "src_problem_idx": src_p,
                    "tgt_problem_idx": tgt_p,
                    "is_repeated_question": bool(src_p == tgt_p),
                    "intervening_interaction_count": n_intervening,
                    "intervening_concept_count": intervening_concept_count,
                    "intervening_same_concept_count": 0,
                    "intervening_category": intervening_category,
                    "delay_interval": interval_label,
                    "legacy_bucket_id": legacy_bucket,
                }
            )

    pairs_df = pd.DataFrame(pairs)
    total_pairs = len(pairs_df)

    # 15-minute grid quantization check
    timestamps = np.concatenate(
        [pairs_df["src_timestamp"].values, pairs_df["tgt_timestamp"].values]
    )
    is_15m_grid = timestamps % 900 == 0
    pct_15m_grid = round(100.0 * float(np.mean(is_15m_grid)), 2)

    forensics = {
        "total_verified_retention_pairs": total_pairs,
        "zero_same_concept_leakage_assert": (same_concept_leakage_count == 0),
        "same_concept_leakage_count": same_concept_leakage_count,
        "negative_or_zero_delta_count": negative_delta_count,
        "timestamp_15min_quantization_pct": pct_15m_grid,
        "isolated_retention_count": int((pairs_df["intervening_category"] == "isolated").sum()),
        "related_interference_count": int(
            (pairs_df["intervening_category"] == "related_interference").sum()
        ),
        "unrelated_interference_count": int(
            (pairs_df["intervening_category"] == "unrelated_interference").sum()
        ),
        "repeated_questions_count": int(pairs_df["is_repeated_question"].sum()),
        "different_questions_count": int((~pairs_df["is_repeated_question"]).sum()),
    }

    logger.info(
        "Extracted %d retention pairs (100%% zero same-concept leakage).",
        total_pairs,
    )
    return pairs_df, forensics


# ==============================================================================
# 2. Model Prediction Alignment & Stratified Evaluation
# ==============================================================================
def evaluate_retention_strata(
    checkpoint_path: Path,
    test_parquet: Path,
    pairs_df: pd.DataFrame,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Run model inference on Test and evaluate Retention AUC / Brier across all strata.
    """
    logger.info("Loading model checkpoint from %s...", checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)

    num_concepts = int(state_dict.get("concept_base_emb", torch.empty(150, 64)).shape[0])
    num_problems = int(state_dict.get("problem_emb.weight", torch.empty(3032, 64)).shape[0])

    model = TPAGKT(
        num_concepts=num_concepts,
        num_problems=num_problems,
        embedding_dim=64,
        hidden_dim=128,
        use_retention_head=True,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Load graph into model if present in test directory
    graph_path = test_parquet.parent / "A_combined.pt"
    if graph_path.exists():
        model.load_graph_from_disk(test_parquet.parent)

    # Collect model predictions aligned by (uuid, orig_idx)
    test_loader = create_dataloader(test_parquet, batch_size=32, shuffle=False)
    pred_map: dict[tuple[str, int], float] = {}

    with torch.no_grad():
        for batch in test_loader:
            out = model(batch)
            ret_mask = batch["ret_mask"].to(device)
            if ret_mask.any():
                pred_ret = out["pred_retention"]
                uuids = batch.get("uuid", ["unknown"] * len(pred_ret))
                orig_idxs = batch.get("orig_idx", torch.full_like(ret_mask, -1))

                for b_i in range(len(pred_ret)):
                    u_str = str(uuids[b_i])
                    m_steps = ret_mask[b_i]
                    if m_steps.any():
                        o_idxs = orig_idxs[b_i, m_steps].cpu().numpy().astype(int)
                        p_vals = pred_ret[b_i, m_steps].cpu().numpy().astype(float)
                        for o_idx, p_val in zip(o_idxs, p_vals, strict=False):
                            pred_map[(u_str, o_idx)] = float(p_val)

    # Attach model predictions to pairs DataFrame
    matched_preds: list[float] = []
    matched_targets: list[int] = []
    valid_indices: list[int] = []

    for idx, row in pairs_df.iterrows():
        key = (row["uuid"], row["src_orig_idx"])
        if key in pred_map:
            matched_preds.append(pred_map[key])
            matched_targets.append(int(row["target_correctness"]))
            valid_indices.append(idx)

    pairs_matched = pairs_df.loc[valid_indices].copy()
    pairs_matched["pred_prob"] = matched_preds
    logger.info(
        "Successfully aligned %d of %d retention predictions (%.1f%%).",
        len(pairs_matched),
        len(pairs_df),
        100.0 * len(pairs_matched) / max(1, len(pairs_df)),
    )

    # Stratified Metric Computation Function
    def _compute_strata_metrics(subset: pd.DataFrame, strata_name: str) -> dict[str, Any]:
        n_samples = len(subset)
        if n_samples == 0:
            return {
                "Stratum": strata_name,
                "Sample Count": 0,
                "Positive Rate": 0.0,
                "Retention AUC": "N/A",
                "Brier Score": "N/A",
                "Macro-F1": "N/A",
            }
        y_t = subset["target_correctness"].values.astype(int)
        y_p = subset["pred_prob"].values.astype(float)
        pos_rate = round(float(np.mean(y_t)), 4)

        if len(set(y_t)) > 1:
            auc_val = round(float(roc_auc_score(y_t, y_p)), 4)
            brier_val = round(float(brier_score_loss(y_t, y_p)), 4)
            pred_bin = (y_p >= 0.50).astype(int)
            f1_val = round(float(f1_score(y_t, pred_bin, average="macro", zero_division=0)), 4)
        else:
            auc_val = "N/A"
            brier_val = round(float(brier_score_loss(y_t, y_p)), 4)
            f1_val = "N/A"

        return {
            "Stratum": strata_name,
            "Sample Count": n_samples,
            "Positive Rate": pos_rate,
            "Retention AUC": auc_val,
            "Brier Score": brier_val,
            "Macro-F1": f1_val,
        }

    strata_rows: list[dict[str, Any]] = []

    # 1. Overall
    strata_rows.append(_compute_strata_metrics(pairs_matched, "Overall Test Retention"))

    # 2. Repeated vs Different Questions
    strata_rows.append(
        _compute_strata_metrics(
            pairs_matched[pairs_matched["is_repeated_question"]], "Repeated Question Only"
        )
    )
    strata_rows.append(
        _compute_strata_metrics(
            pairs_matched[~pairs_matched["is_repeated_question"]], "Different Question Only"
        )
    )

    # 3. Intervening Practice Categories
    for cat, label in [
        ("isolated", "Group A: Isolated Retention (0 Intervening)"),
        ("related_interference", "Group B: Related Interference (Graph Neighbors)"),
        ("unrelated_interference", "Group C: Unrelated Interference (No Graph Edge)"),
    ]:
        strata_rows.append(
            _compute_strata_metrics(
                pairs_matched[pairs_matched["intervening_category"] == cat], label
            )
        )

    # 4. Fine-Grained Delay Intervals
    for interval, label in [
        ("<15m", "Delay <= 15 min (Quantized Immediate)"),
        ("15m-60m", "Delay 15m - 60m (Within Session)"),
        ("1h-24h", "Delay 1h - 24h (Same-Day Review)"),
        ("1d-7d", "Delay 1d - 7d (Multi-Day Spaced Review)"),
        (">7d", "Delay > 7 days (Long-Term Retention)"),
    ]:
        strata_rows.append(
            _compute_strata_metrics(
                pairs_matched[pairs_matched["delay_interval"] == interval], label
            )
        )

    # 5. Cross-Tabulation: Delay Interval x Intervening Practice
    for interval in ["<15m", "15m-60m", "1h-24h", "1d-7d", ">7d"]:
        for cat in ["isolated", "related_interference", "unrelated_interference"]:
            sub = pairs_matched[
                (pairs_matched["delay_interval"] == interval)
                & (pairs_matched["intervening_category"] == cat)
            ]
            if len(sub) > 0:
                strata_rows.append(_compute_strata_metrics(sub, f"{interval} | {cat}"))

    metrics_df = pd.DataFrame(strata_rows)
    report_dict = {
        "total_pairs_evaluated": len(pairs_matched),
        "overall_retention_auc": strata_rows[0]["Retention AUC"],
        "overall_brier_score": strata_rows[0]["Brier Score"],
        "strata_results": strata_rows,
    }

    return report_dict, metrics_df


# ==============================================================================
# 3. Report Generator & CLI Entrypoint
# ==============================================================================
def run_retention_audit_pipeline(
    data_dir: Path,
    checkpoints_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> None:
    """Execute complete Phase 2 audit pipeline and save outputs cleanly."""
    output_dir.mkdir(parents=True, exist_ok=True)
    test_parquet = data_dir / "test.parquet"
    graph_path = data_dir / "A_combined.pt"

    # 1. Forensic Extraction
    pairs_df, forensics = extract_and_verify_retention_pairs(test_parquet, graph_path)

    # Save reconstructed pairs
    pairs_parquet_path = output_dir / "retention_pairs_forensic.parquet"
    pairs_df.to_parquet(pairs_parquet_path, index=False)
    logger.info("Saved forensic pairs to %s", pairs_parquet_path)

    # 2. Checkpoint Selection for Evaluation
    candidate_checkpoints = [
        checkpoints_dir / "tpa_gkt_full_seed42_best.pt",
        checkpoints_dir / "tpa_gkt_full_seed42_fastdev_best.pt",
    ]
    selected_checkpoint = None
    for cp in candidate_checkpoints:
        if cp.exists():
            selected_checkpoint = cp
            break

    if selected_checkpoint is None:
        logger.warning(
            "No TPA-GKT checkpoint found in %s. Skipping model evaluation.", checkpoints_dir
        )
        metrics_df = pd.DataFrame()
        eval_report = {}
    else:
        eval_report, metrics_df = evaluate_retention_strata(
            selected_checkpoint, test_parquet, pairs_df, device
        )

    # 3. Save JSON Report
    combined_report = {
        "forensics": forensics,
        "evaluation": eval_report,
    }
    json_path = output_dir / "retention_forensic_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(combined_report, f, indent=2)
    logger.info("Saved JSON forensic report to %s", json_path)

    # 4. Save CSV Table
    if not metrics_df.empty:
        csv_path = output_dir / "retention_stratified_metrics.csv"
        metrics_df.to_csv(csv_path, index=False)
        logger.info("Saved stratified metrics to %s", csv_path)

    # 5. Generate Markdown Summary
    md_path = output_dir / "retention_audit_summary.md"
    n_pairs = forensics["total_verified_retention_pairs"]
    leak_status = (
        "100% (VERIFIED)" if forensics["zero_same_concept_leakage_assert"] else "VIOLATED"
    )
    t_quant = forensics["timestamp_15min_quantization_pct"]
    n_iso = forensics["isolated_retention_count"]
    n_rel = forensics["related_interference_count"]
    n_unrel = forensics["unrelated_interference_count"]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 2 Retention Target & Temporal Leakage Forensic Audit\n\n")
        f.write(f"- **Total Verified Test Retention Pairs**: {n_pairs:,}\n")
        f.write(f"- **Zero Same-Concept Leakage Compliance**: {leak_status}\n")
        f.write(f"- **15-Minute Grid Timestamp Quantization**: {t_quant}%\n\n")
        f.write("## Intervening Practice Disentanglement\n\n")
        f.write(f"1. **Isolated Retention (0 Intervening Practice)**: {n_iso:,} pairs\n")
        f.write(f"2. **Related-Concept Interference (Graph Neighbors)**: {n_rel:,} pairs\n")
        f.write(f"3. **Unrelated-Concept Interference (Unconnected)**: {n_unrel:,} pairs\n\n")
        if not metrics_df.empty:
            f.write("## Stratified Metric Performance\n\n")
            f.write(metrics_df.to_markdown(index=False) + "\n")
    logger.info("Saved markdown summary to %s", md_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 Retention Target & Leakage Audit")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--checkpoints-dir", type=Path, default=Path("services/gnn_kt/outputs/checkpoints")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("services/gnn_kt/outputs/validation_fixed/retention_audit"),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_retention_audit_pipeline(args.data_dir, args.checkpoints_dir, args.output_dir, device)


if __name__ == "__main__":
    main()
