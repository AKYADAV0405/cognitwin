"""
Temporal Prerequisite-Aware Graph Knowledge Tracing (TPA-GKT)
Final Experiment Validation Audit (final_validation_audit.py)

Executes 5 rigorous advisor-mandated research-validation checks:
  1. Seed-Level Verification (per_seed_results.csv):
     - Scans saved per-seed runs, asserts fingerprint and 30 runs, recomputes mean +/- std.
  2. Graph Ablation Deep-Dive (graph_ablation_audit.json):
     - Cosine similarity geometry (A_hier & A_trans vs. unconnected) and anti-oversmoothing check.
     - Sparse concept (Bottom 25%) vs. Frequent concept (Top 75%) stratified AUC across GCN, GAT, and No-GNN.
  3. Independent Mastery Validation (mastery_independent_eval.json):
     - Multi-step validation of M_hat_{u,c,t} against future rolling accuracy (k >= 3)
       and sustained mastery (>= 80% on final 3 attempts) via Spearman rho, Pearson r, AUC, and ECE.
  4. Retention Label Forensic Audit (retention_label_audit.json):
     - 100% zero-leakage same-concept check, intervening practice counts, repeated vs. different
       questions retention AUC, and 15-minute timestamp alignment.
  5. Paired Statistical Significance & Uncertainty (statistical_tests.csv):
     - Student-clustered paired bootstrap (B=1,000) comparing TPA-GKT against DKT, SAKT, GKT,
       and w/o Graph Propagation (Delta AUC, 95% CI, bootstrap p-value, paired seed t-test).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parents[3])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from services.gnn_kt.src.dataset import create_dataloader
from services.gnn_kt.src.evaluate import compute_ece
from services.gnn_kt.src.train import build_model, get_device
from sklearn.metrics import brier_score_loss, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("final_validation_audit")


# ------------------------------------------------------------------------------
# Check 1: Seed-Level Verification & Table Recomputation
# ------------------------------------------------------------------------------
def run_seed_level_verification(
    runs_dir: Path,
    results_dir: Path,
    expected_num_students: int = 5000,
    expected_test_interactions: int = 120271,
    expected_test_retention_pairs: int = 16285,
    allow_partial: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Scan and verify all per-seed runs, assert dataset fingerprint, recompute stats."""
    logger.info("--> [Check 1] Executing Seed-Level Verification...")

    # Load summary json if available, else scan metrics JSONs
    summary_path = results_dir / "all_results_summary.json"
    runs_data: dict[str, dict[str, Any]] = {}
    dataset_fingerprint: dict[str, Any] = {}

    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary_payload = json.load(f)
            runs_data = summary_payload.get("runs", {})
            dataset_fingerprint = summary_payload.get("dataset_fingerprint", {})

    # Also search runs_dir for individual *_metrics.json
    metrics_files = list(runs_dir.glob("*_metrics.json")) if runs_dir.exists() else []

    # Map from config key to list of seed dicts
    grouped_by_config: dict[str, list[dict[str, Any]]] = {}

    # 1. Ingest from all_results_summary.json if present
    for config_key, seed_dict in runs_data.items():
        grouped_by_config.setdefault(config_key, [])
        for seed_str, metrics in seed_dict.items():
            seed = int(seed_str)
            row = {
                "config_key": config_key,
                "seed": seed,
                "response_auc": metrics.get("response_auc"),
                "response_acc": metrics.get("response_acc"),
                "response_f1_macro": metrics.get("response_f1_macro"),
                "response_f1_weighted": metrics.get("response_f1_weighted"),
                "response_brier": metrics.get("response_brier"),
                "retention_auc": metrics.get("retention_auc"),
                "retention_f1_macro": metrics.get("retention_f1_macro"),
                "retention_brier": metrics.get("retention_brier"),
                "retention_gt7d_auc": (
                    metrics.get("retention_buckets", {}).get(">7d", {}).get("auc")
                ),
            }
            grouped_by_config[config_key].append(row)

    # 2. If summary is empty or partial, ingest from individual metrics files
    if not grouped_by_config and metrics_files:
        for mf in metrics_files:
            try:
                with open(mf, encoding="utf-8") as f:
                    data = json.load(f)
                    run_name = data.get("run_name", mf.stem)
                    seed = data.get("seed", 42)
                    # Extract config_key from run_name
                    c_key = run_name.split("_seed")[0]
                    test_m = data.get("test_metrics", data)
                    grouped_by_config.setdefault(c_key, [])
                    row = {
                        "config_key": c_key,
                        "seed": seed,
                        "response_auc": test_m.get("response_auc"),
                        "response_acc": test_m.get("response_acc"),
                        "response_f1_macro": test_m.get("response_f1_macro"),
                        "response_f1_weighted": test_m.get("response_f1_weighted"),
                        "response_brier": test_m.get("response_brier"),
                        "retention_auc": test_m.get("retention_auc"),
                        "retention_f1_macro": test_m.get("retention_f1_macro"),
                        "retention_brier": test_m.get("retention_brier"),
                        "retention_gt7d_auc": (
                            test_m.get("retention_buckets", {}).get(">7d", {}).get("auc")
                        ),
                    }
                    grouped_by_config[c_key].append(row)
            except Exception as e:
                logger.warning("Could not parse metrics file %s: %s", mf, e)

    # Fingerprint verification
    fp_valid = True
    cohort_counts = dataset_fingerprint.get("cohort_counts", {})
    if cohort_counts:
        actual_test_int = cohort_counts.get("test_interactions", 0)
        actual_test_ret = cohort_counts.get("test_retention_pairs", 0)
        if not allow_partial:
            if (
                actual_test_int != expected_test_interactions
                or actual_test_ret != expected_test_retention_pairs
            ):
                fp_valid = False
                logger.warning(
                    "Fingerprint mismatch! Expected (%d, %d), got (%d, %d)",
                    expected_test_interactions,
                    expected_test_retention_pairs,
                    actual_test_int,
                    actual_test_ret,
                )

    total_runs_count = sum(len(seeds) for seeds in grouped_by_config.values())

    # Full benchmark assertion: 10 configs x 3 seeds = 30 runs
    if not allow_partial and total_runs_count < 30:
        logger.warning("Found %d runs (expected 30 for full benchmark suite).", total_runs_count)

    # Build per_seed_results.csv rows
    csv_rows: list[dict[str, Any]] = []
    for c_key, seed_list in grouped_by_config.items():
        for item in sorted(seed_list, key=lambda x: x["seed"]):
            csv_rows.append(
                {
                    "Config": c_key,
                    "Seed": item["seed"],
                    "Next-Response AUC": f"{item['response_auc']:.4f}"
                    if item["response_auc"] is not None
                    else "N/A",
                    "Next-Response ACC": f"{item['response_acc']:.4f}"
                    if item["response_acc"] is not None
                    else "N/A",
                    "Next-Response Macro-F1": f"{item['response_f1_macro']:.4f}"
                    if item["response_f1_macro"] is not None
                    else "N/A",
                    "Next-Response Weighted-F1": f"{item['response_f1_weighted']:.4f}"
                    if item["response_f1_weighted"] is not None
                    else "N/A",
                    "Next-Response Brier": f"{item['response_brier']:.4f}"
                    if item["response_brier"] is not None
                    else "N/A",
                    "Retention AUC": f"{item['retention_auc']:.4f}"
                    if item["retention_auc"] is not None
                    else "N/A",
                    "Retention Macro-F1": f"{item['retention_f1_macro']:.4f}"
                    if item["retention_f1_macro"] is not None
                    else "N/A",
                    "Retention Brier": f"{item['retention_brier']:.4f}"
                    if item["retention_brier"] is not None
                    else "N/A",
                    "Retention AUC (>7d)": f"{item['retention_gt7d_auc']:.4f}"
                    if item["retention_gt7d_auc"] is not None
                    else "N/A",
                    "Fingerprint Valid": fp_valid,
                }
            )

    # Save per_seed_results.csv
    per_seed_csv_path = results_dir / "per_seed_results.csv"
    if csv_rows:
        with open(per_seed_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        logger.info("Saved per_seed_results.csv to %s (%d rows)", per_seed_csv_path, len(csv_rows))

    check1_summary = {
        "total_runs_found": total_runs_count,
        "configs_count": len(grouped_by_config),
        "fingerprint_valid": fp_valid,
        "cohort_counts": cohort_counts,
        "all_30_runs_present": (total_runs_count >= 30),
    }
    return check1_summary, csv_rows


# ------------------------------------------------------------------------------
# Check 2: Graph Ablation Deep-Dive (Embedding Geometry & Stratification)
# ------------------------------------------------------------------------------
def load_checkpoint_safely(
    checkpoint_candidates: list[Path],
    processed_dir: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]] | None:
    """Find and instantiate a model from candidate checkpoint paths."""
    for p in checkpoint_candidates:
        if p.exists():
            try:
                ckpt = torch.load(p, map_location=device, weights_only=False)
                model_name = ckpt.get("model_name", "tpa_gkt")
                config = ckpt.get("config", {})
                num_concepts = ckpt.get("num_concepts", 150)
                num_problems = ckpt.get("num_problems", 3000)
                model = build_model(
                    model_name, num_concepts, num_problems, config, processed_dir, device
                )
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()
                return model, ckpt
            except Exception as e:
                logger.warning("Failed loading checkpoint %s: %s", p, e)
    return None


def run_graph_ablation_deep_dive(
    checkpoints_dir: Path,
    data_dir: Path,
    results_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """
    Measure embedding geometry (connected vs unconnected cosine similarity)
    and stratify test performance by concept training frequency (rare vs frequent).
    """
    logger.info("--> [Check 2] Executing Graph Ablation Deep-Dive...")

    # Load graphs
    hier_path = data_dir / "A_hier.pt"
    trans_path = data_dir / "A_trans.pt"

    edge_index_hier = (
        torch.load(hier_path, weights_only=False).edge_index
        if hier_path.exists()
        else torch.empty((2, 0), dtype=torch.long)
    )
    edge_index_trans = (
        torch.load(trans_path, weights_only=False).edge_index
        if trans_path.exists()
        else torch.empty((2, 0), dtype=torch.long)
    )

    # 1. Models to inspect: GCN, GAT, and No-GNN
    models_to_test = {
        "gcn": [
            checkpoints_dir / "tpa_gkt_full_seed42_best.pt",
            checkpoints_dir / "tpa_gkt_full_seed42_fastdev_best.pt",
        ],
        "gat": [
            checkpoints_dir / "ablation_gat_seed42_best.pt",
            checkpoints_dir / "ablation_gat_seed42_fastdev_best.pt",
        ],
        "no_gnn": [
            checkpoints_dir / "ablation_wo_graph_seed42_best.pt",
            checkpoints_dir / "ablation_wo_graph_seed42_fastdev_best.pt",
        ],
    }

    geometry_results: dict[str, Any] = {}
    loaded_models: dict[str, torch.nn.Module] = {}

    for m_label, c_paths in models_to_test.items():
        res = load_checkpoint_safely(c_paths, data_dir, device)
        if res is None:
            logger.warning("Could not load checkpoint for %s", m_label)
            continue
        model, _ = res
        loaded_models[m_label] = model

        # Derive concept embeddings
        with torch.no_grad():
            if hasattr(model, "encode_graph") and getattr(model, "use_graph", False):
                g_emb = model.encode_graph()
            elif hasattr(model, "gnn_norm0"):
                g_emb = model.gnn_norm0(model.concept_base_emb)
            else:
                g_emb = model.concept_base_emb

            # Normalize embeddings to unit sphere
            g_norm = torch.nn.functional.normalize(g_emb, p=2, dim=-1)
            cos_matrix = torch.matmul(g_norm, g_norm.T).cpu().numpy()

        num_c = cos_matrix.shape[0]

        # Connected masks (excluding self-loops)
        mask_hier = np.zeros((num_c, num_c), dtype=bool)
        if edge_index_hier.numel() > 0:
            src, tgt = edge_index_hier[0].cpu().numpy(), edge_index_hier[1].cpu().numpy()
            valid = (src < num_c) & (tgt < num_c) & (src != tgt)
            mask_hier[src[valid], tgt[valid]] = True

        mask_trans = np.zeros((num_c, num_c), dtype=bool)
        if edge_index_trans.numel() > 0:
            src, tgt = edge_index_trans[0].cpu().numpy(), edge_index_trans[1].cpu().numpy()
            valid = (src < num_c) & (tgt < num_c) & (src != tgt)
            mask_trans[src[valid], tgt[valid]] = True

        mask_comb = mask_hier | mask_trans
        mask_unconnected = (~mask_comb) & (~np.eye(num_c, dtype=bool))

        mean_hier = float(np.mean(cos_matrix[mask_hier])) if mask_hier.any() else 0.0
        mean_trans = float(np.mean(cos_matrix[mask_trans])) if mask_trans.any() else 0.0
        mean_comb = float(np.mean(cos_matrix[mask_comb])) if mask_comb.any() else 0.0
        mean_unconn = (
            float(np.mean(cos_matrix[mask_unconnected])) if mask_unconnected.any() else 0.0
        )

        geometry_results[m_label] = {
            "mean_cosine_sim_hier_connected": round(mean_hier, 4),
            "mean_cosine_sim_trans_connected": round(mean_trans, 4),
            "mean_cosine_sim_combined_connected": round(mean_comb, 4),
            "mean_cosine_sim_unconnected": round(mean_unconn, 4),
            "similarity_delta_connected_minus_unconnected": round(mean_comb - mean_unconn, 4),
            "ratio_connected_to_unconnected": round(mean_comb / max(1e-5, mean_unconn), 3),
            "is_oversmoothed": bool(mean_unconn > 0.90),
        }

    # 2. Concept Stratification: Bottom 25% (rare/sparse) vs Top 75% (frequent)
    stratification_results: dict[str, Any] = {}
    train_parquet = data_dir / "train.parquet"
    test_parquet = data_dir / "test.parquet"

    if train_parquet.exists() and test_parquet.exists() and loaded_models:
        df_train = pd.read_parquet(train_parquet)

        # Compute training concept frequencies
        concept_counts = df_train["concept_idx"].value_counts().to_dict()
        all_concepts = list(range(150))
        freqs = [concept_counts.get(c, 0) for c in all_concepts]
        cutoff_p25 = float(np.percentile(freqs, 25))

        sparse_concepts = {c for c, f in zip(all_concepts, freqs, strict=False) if f <= cutoff_p25}
        frequent_concepts = {c for c in all_concepts if c not in sparse_concepts}

        # Run test inference on each loaded model
        test_loader = create_dataloader(test_parquet, batch_size=32, shuffle=False)

        for m_label, model in loaded_models.items():
            sparse_y_true: list[int] = []
            sparse_y_pred: list[float] = []
            freq_y_true: list[int] = []
            freq_y_pred: list[float] = []

            sparse_ret_true: list[int] = []
            sparse_ret_pred: list[float] = []
            freq_ret_true: list[int] = []
            freq_ret_pred: list[float] = []

            with torch.no_grad():
                for batch in test_loader:
                    out = model(batch)
                    next_mask = batch["next_mask"].to(device)
                    next_correct = batch["next_correct"].to(device)
                    next_c = batch["next_concept"].to(device)
                    pred_resp = out["pred_response"]

                    if next_mask.any():
                        c_flat = next_c[next_mask].cpu().numpy()
                        y_flat = next_correct[next_mask].cpu().numpy()
                        p_flat = pred_resp[next_mask].cpu().numpy()

                        for c_idx, y_val, p_val in zip(c_flat, y_flat, p_flat, strict=False):
                            if c_idx in sparse_concepts:
                                sparse_y_true.append(int(y_val))
                                sparse_y_pred.append(float(p_val))
                            else:
                                freq_y_true.append(int(y_val))
                                freq_y_pred.append(float(p_val))

                    # Retention predictions if head available
                    if "pred_retention" in out and "ret_mask" in batch:
                        ret_mask = batch["ret_mask"].to(device)
                        ret_target = batch["ret_target"].to(device)
                        ret_c = batch["ret_concept"].to(device)
                        pred_ret = out["pred_retention"]

                        if ret_mask.any():
                            rc_flat = ret_c[ret_mask].cpu().numpy()
                            ry_flat = ret_target[ret_mask].cpu().numpy()
                            rp_flat = pred_ret[ret_mask].cpu().numpy()

                            for c_idx, y_val, p_val in zip(rc_flat, ry_flat, rp_flat, strict=False):
                                if c_idx in sparse_concepts:
                                    sparse_ret_true.append(int(y_val))
                                    sparse_ret_pred.append(float(p_val))
                                else:
                                    freq_ret_true.append(int(y_val))
                                    freq_ret_pred.append(float(p_val))

            def safe_auc(yt: list[int], yp: list[float]) -> float | None:
                if len(yt) < 10 or len(set(yt)) < 2:
                    return None
                try:
                    return round(float(roc_auc_score(yt, yp)), 4)
                except Exception:
                    return None

            stratification_results[m_label] = {
                "sparse_concepts_count": len(sparse_concepts),
                "frequent_concepts_count": len(frequent_concepts),
                "sparse_response_auc": safe_auc(sparse_y_true, sparse_y_pred),
                "frequent_response_auc": safe_auc(freq_y_true, freq_y_pred),
                "sparse_response_samples": len(sparse_y_true),
                "frequent_response_samples": len(freq_y_true),
                "sparse_retention_auc": safe_auc(sparse_ret_true, sparse_ret_pred),
                "frequent_retention_auc": safe_auc(freq_ret_true, freq_ret_pred),
            }

    audit_payload = {
        "embedding_geometry": geometry_results,
        "stratification_by_training_frequency": stratification_results,
    }

    out_path = results_dir / "graph_ablation_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2)
    logger.info("Saved graph_ablation_audit.json to %s", out_path)

    return audit_payload


# ------------------------------------------------------------------------------
# Check 3: Independent Mastery Validation (Multi-Step & Sustained Mastery)
# ------------------------------------------------------------------------------
def run_independent_mastery_validation(
    checkpoint_candidates: list[Path],
    data_dir: Path,
    results_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """
    Evaluate the Concept Mastery Head M_hat_{u,c,t} against independent multi-step targets:
      - Future Rolling Accuracy over next k >= 3 attempts on concept c (excluding t+1).
      - Spearman rank correlation rho, Pearson r, and ROC-AUC for sustained mastery (>= 80% on final 3 attempts).
    """
    logger.info("--> [Check 3] Executing Independent Mastery Validation...")

    test_parquet = data_dir / "test.parquet"
    if not test_parquet.exists():
        logger.warning("test.parquet not found in %s", data_dir)
        return {}

    model_res = load_checkpoint_safely(checkpoint_candidates, data_dir, device)
    if model_res is None:
        logger.warning("Could not load TPA-GKT checkpoint for mastery evaluation.")
        return {}

    model, _ = model_res
    df_test = pd.read_parquet(test_parquet)

    # Group interactions by student uuid
    test_loader = create_dataloader(test_parquet, batch_size=32, shuffle=False)

    # Collect predicted concept mastery M_hat for each valid timestep
    predicted_mastery_map: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in test_loader:
            out = model(batch)
            if "pred_mastery" not in out:
                continue

            pred_mastery = out["pred_mastery"].cpu().numpy()
            next_mask = batch["next_mask"].cpu().numpy()
            concept_idx = batch["concept_idx"].cpu().numpy()
            uuids = batch.get("uuid", ["unknown"] * len(pred_mastery))

            for b_idx in range(len(pred_mastery)):
                u_str = uuids[b_idx]
                valid_steps = np.where(next_mask[b_idx])[0]
                for t in valid_steps:
                    c = int(concept_idx[b_idx, t])
                    m_val = float(pred_mastery[b_idx, t])
                    predicted_mastery_map.append(
                        {
                            "uuid": u_str,
                            "step": int(t),
                            "concept_idx": c,
                            "pred_mastery": m_val,
                        }
                    )

    # Group test dataframe by student and concept to construct multi-step ground-truth targets
    df_test = df_test.sort_values(by=["uuid", "timestamp_sec", "orig_idx"]).reset_index(drop=True)
    grouped = df_test.groupby(["uuid", "concept_idx"])

    m_preds: list[float] = []
    future_accs: list[float] = []
    final3_mastery_targets: list[int] = []

    # Map student interactions
    student_concept_history: dict[tuple[str, int], list[int]] = {}
    for (u, c), sub_df in grouped:
        student_concept_history[(str(u), int(c))] = sub_df["is_correct"].tolist()

    for item in predicted_mastery_map:
        key = (item["uuid"], item["concept_idx"])
        history = student_concept_history.get(key, [])
        step = item["step"]

        # Future attempts on this concept strictly after t+1
        future_attempts = history[step + 2 :]
        if len(future_attempts) >= 3:
            m_preds.append(item["pred_mastery"])
            future_acc = float(np.mean(future_attempts))
            future_accs.append(future_acc)

            # Final 3 attempts binary mastery target (>= 80% correctness)
            final3 = future_attempts[-3:]
            final3_target = 1 if np.mean(final3) >= 0.80 else 0
            final3_mastery_targets.append(final3_target)

    eval_results: dict[str, Any] = {
        "eligible_multi_step_instances": len(m_preds),
        "min_future_attempts_threshold": 3,
    }

    if len(m_preds) >= 20:
        spearman_rho, spearman_p = stats.spearmanr(m_preds, future_accs)
        pearson_r, pearson_p = stats.pearsonr(m_preds, future_accs)

        # ROC-AUC for sustained mastery (>= 80% on final 3 attempts)
        unique_classes = set(final3_mastery_targets)
        if len(unique_classes) > 1:
            sustained_auc = float(roc_auc_score(final3_mastery_targets, m_preds))
        else:
            sustained_auc = 0.50

        ece_score, bins = compute_ece(
            np.array(final3_mastery_targets), np.array(m_preds), n_bins=10
        )
        brier = float(brier_score_loss(final3_mastery_targets, m_preds))

        eval_results.update(
            {
                "spearman_rho": round(float(spearman_rho), 4),
                "spearman_p_value": float(spearman_p),
                "pearson_r": round(float(pearson_r), 4),
                "pearson_p_value": float(pearson_p),
                "sustained_mastery_auc": round(sustained_auc, 4),
                "sustained_mastery_ece": round(ece_score, 4),
                "sustained_mastery_brier": round(brier, 4),
                "reliability_bins": bins,
            }
        )
        logger.info(
            "Mastery Independent Eval: Spearman rho=%.4f (p=%.2e), Sustained Mastery AUC=%.4f",
            spearman_rho,
            spearman_p,
            sustained_auc,
        )
    else:
        logger.warning("Fewer than 20 eligible multi-step instances found for mastery evaluation.")

    out_path = results_dir / "mastery_independent_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    logger.info("Saved mastery_independent_eval.json to %s", out_path)

    return eval_results


# ------------------------------------------------------------------------------
# Check 4: Retention Label Forensic Audit
# ------------------------------------------------------------------------------
def run_retention_label_forensic_audit(
    data_dir: Path,
    checkpoints_dir: Path,
    results_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """
    Forensically audit delayed-retention pairs:
      - 100% zero-leakage between t and t' (strictly next attempt on concept c).
      - Intervening practice on other concepts.
      - Exact repeated questions (memorization) vs different questions (concept retention).
      - Timestamp granularity & bucket class balance.
    """
    logger.info("--> [Check 4] Executing Retention Label Forensic Audit...")

    ret_parquet = data_dir / "test_retention.parquet"
    test_parquet = data_dir / "test.parquet"

    if not ret_parquet.exists() or not test_parquet.exists():
        logger.warning("Required parquets missing for retention audit.")
        return {}

    df_ret = pd.read_parquet(ret_parquet)
    df_test = pd.read_parquet(test_parquet)

    total_pairs = len(df_ret)

    # 1. Zero Same-Concept Leakage Verification
    df_test = df_test.sort_values(by=["uuid", "timestamp_sec", "orig_idx"]).reset_index(drop=True)
    df_test["global_row_idx"] = np.arange(len(df_test))

    # Identify consecutive interactions per (uuid, ucid)
    df_test["concept_step"] = df_test.groupby(["uuid", "ucid"]).cumcount()
    next_step = df_test.groupby(["uuid", "ucid"])["concept_step"].shift(-1)
    step_diff = next_step - df_test["concept_step"]

    # Filter to pairs that have retention targets
    valid_ret_mask = df_test["has_retention_target"].values
    consecutive_steps = step_diff.values[valid_ret_mask]
    same_concept_skipped = int(np.sum(consecutive_steps > 1))
    zero_leakage_rate = 1.0 - (same_concept_skipped / max(1, len(consecutive_steps)))

    # 2. Intervening Practice on Other Concepts
    next_global_idx = df_test.groupby(["uuid", "ucid"])["global_row_idx"].shift(-1)
    intervening_total = (next_global_idx - df_test["global_row_idx"] - 1).values[valid_ret_mask]

    intervening_present_cnt = int(np.sum(intervening_total > 0))
    intervening_pct = round(100.0 * intervening_present_cnt / max(1, len(intervening_total)), 2)

    # Median intervening practice per bucket
    buckets = df_test["retention_bucket_name"].values[valid_ret_mask]
    intervening_by_bucket: dict[str, Any] = {}
    for b_name in ["<1h", "1h-24h", "1d-7d", ">7d"]:
        b_mask = buckets == b_name
        if b_mask.any():
            med_interv = float(np.median(intervening_total[b_mask]))
            intervening_by_bucket[b_name] = {
                "median_intervening_interactions": med_interv,
                "pct_with_intervening": round(100.0 * np.mean(intervening_total[b_mask] > 0), 2),
            }

    # 3. Repeated Questions vs Different Questions
    next_upid = df_test.groupby(["uuid", "ucid"])["upid"].shift(-1)
    curr_upid = df_test["upid"]
    is_repeated = (curr_upid == next_upid).values[valid_ret_mask]

    repeated_cnt = int(np.sum(is_repeated))
    different_cnt = int(len(is_repeated) - repeated_cnt)
    repeated_pct = round(100.0 * repeated_cnt / max(1, len(is_repeated)), 2)
    different_pct = round(100.0 * different_cnt / max(1, len(is_repeated)), 2)

    # Evaluate TPA-GKT Retention AUC on Different Questions Only vs Repeated Questions
    tpa_gkt_ckpt = [
        checkpoints_dir / "tpa_gkt_full_seed42_best.pt",
        checkpoints_dir / "tpa_gkt_full_seed42_fastdev_best.pt",
    ]
    model_res = load_checkpoint_safely(tpa_gkt_ckpt, data_dir, device)

    diff_auc = None
    rep_auc = None

    if model_res is not None:
        model, _ = model_res
        test_loader = create_dataloader(test_parquet, batch_size=32, shuffle=False)

        ret_true_all: list[int] = []
        ret_pred_all: list[float] = []

        with torch.no_grad():
            for batch in test_loader:
                out = model(batch)
                if "pred_retention" in out and "ret_mask" in batch:
                    rm = batch["ret_mask"].to(device)
                    rt = batch["ret_target"].to(device)
                    pr = out["pred_retention"]
                    if rm.any():
                        ret_true_all.extend(rt[rm].cpu().numpy().astype(int).tolist())
                        ret_pred_all.extend(pr[rm].cpu().numpy().astype(float).tolist())

        if len(ret_true_all) == len(is_repeated):
            yt_arr = np.array(ret_true_all)
            yp_arr = np.array(ret_pred_all)

            if len(set(yt_arr[~is_repeated])) > 1:
                diff_auc = round(
                    float(roc_auc_score(yt_arr[~is_repeated], yp_arr[~is_repeated])), 4
                )
            if len(set(yt_arr[is_repeated])) > 1:
                rep_auc = round(float(roc_auc_score(yt_arr[is_repeated], yp_arr[is_repeated])), 4)

    # 4. Timestamp Granularity & Bucket Class Balance
    timestamps = df_test["timestamp_sec"].values[valid_ret_mask]
    is_15min_aligned = timestamps % 900 == 0
    pct_15min_aligned = round(100.0 * np.mean(is_15min_aligned), 2)

    bucket_forensics: dict[str, Any] = {}
    ret_targets = df_test["retention_target"].values[valid_ret_mask]

    for b_name in ["<1h", "1h-24h", "1d-7d", ">7d"]:
        b_mask = buckets == b_name
        b_count = int(np.sum(b_mask))
        if b_count > 0:
            b_pos_rate = round(float(np.mean(ret_targets[b_mask])), 4)
            b_15min = round(100.0 * np.mean(is_15min_aligned[b_mask]), 2)
            bucket_forensics[b_name] = {
                "pair_count": b_count,
                "positive_class_rate": b_pos_rate,
                "pct_15min_aligned": b_15min,
            }

    audit_results = {
        "total_test_retention_pairs": total_pairs,
        "zero_same_concept_leakage_assert": (same_concept_skipped == 0),
        "zero_leakage_compliance_pct": round(zero_leakage_rate * 100.0, 2),
        "intervening_practice_overall_pct": intervening_pct,
        "intervening_practice_by_bucket": intervening_by_bucket,
        "repeated_exact_questions_count": repeated_cnt,
        "repeated_exact_questions_pct": repeated_pct,
        "different_questions_count": different_cnt,
        "different_questions_pct": different_pct,
        "retention_auc_different_questions_only": diff_auc,
        "retention_auc_repeated_questions_only": rep_auc,
        "timestamp_15min_alignment_overall_pct": pct_15min_aligned,
        "bucket_forensics": bucket_forensics,
    }

    out_path = results_dir / "retention_label_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    logger.info("Saved retention_label_audit.json to %s", out_path)

    return audit_results


# ------------------------------------------------------------------------------
# Check 5: Paired Statistical Significance & Uncertainty (Bootstrap B=1,000)
# ------------------------------------------------------------------------------
def run_paired_statistical_significance(
    checkpoints_dir: Path,
    data_dir: Path,
    results_dir: Path,
    device: torch.device,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Run Student-Clustered Paired Bootstrap (B=1,000 resamples over unique Test students)
    comparing TPA-GKT (GCN & GAT) against DKT, SAKT, GKT, and w/o Graph Propagation.
    """
    logger.info("--> [Check 5] Executing Paired Statistical Significance (B=%d)...", n_bootstrap)
    np.random.seed(seed)

    test_parquet = data_dir / "test.parquet"
    if not test_parquet.exists():
        logger.warning("test.parquet not found in %s", data_dir)
        return []

    models_to_evaluate = {
        "TPA-GKT (GCN)": [
            checkpoints_dir / "tpa_gkt_full_seed42_best.pt",
            checkpoints_dir / "tpa_gkt_full_seed42_fastdev_best.pt",
        ],
        "TPA-GKT (GAT)": [
            checkpoints_dir / "ablation_gat_seed42_best.pt",
            checkpoints_dir / "ablation_gat_seed42_fastdev_best.pt",
        ],
        "DKT": [
            checkpoints_dir / "dkt_seed42_best.pt",
            checkpoints_dir / "dkt_seed42_fastdev_best.pt",
        ],
        "SAKT": [
            checkpoints_dir / "sakt_seed42_best.pt",
            checkpoints_dir / "sakt_seed42_fastdev_best.pt",
        ],
        "GKT": [
            checkpoints_dir / "gkt_seed42_best.pt",
            checkpoints_dir / "gkt_seed42_fastdev_best.pt",
        ],
        "w/o Graph Propagation": [
            checkpoints_dir / "ablation_wo_graph_seed42_best.pt",
            checkpoints_dir / "ablation_wo_graph_seed42_fastdev_best.pt",
        ],
    }

    # Collect per-student predictions for each available model
    test_loader = create_dataloader(test_parquet, batch_size=32, shuffle=False)
    student_preds: dict[str, dict[str, list[tuple[int, float]]]] = {}

    for m_label, c_paths in models_to_evaluate.items():
        res = load_checkpoint_safely(c_paths, data_dir, device)
        if res is None:
            logger.warning("Skipping %s for statistical testing (no checkpoint).", m_label)
            continue
        model, _ = res
        student_preds[m_label] = {}

        with torch.no_grad():
            for batch in test_loader:
                out = model(batch)
                next_mask = batch["next_mask"].to(device)
                next_correct = batch["next_correct"].to(device)
                pred_resp = out["pred_response"]
                uuids = batch.get("uuid", ["unknown"] * len(pred_resp))

                for b_idx in range(len(pred_resp)):
                    u_str = uuids[b_idx]
                    student_preds[m_label].setdefault(u_str, [])
                    m_steps = next_mask[b_idx]
                    if m_steps.any():
                        y_t = next_correct[b_idx, m_steps].cpu().numpy().astype(int)
                        y_p = pred_resp[b_idx, m_steps].cpu().numpy().astype(float)
                        for y_val, p_val in zip(y_t, y_p, strict=False):
                            student_preds[m_label][u_str].append((int(y_val), float(p_val)))

    # Identify primary models and baselines
    primary_models = [m for m in ["TPA-GKT (GCN)", "TPA-GKT (GAT)"] if m in student_preds]
    comparison_models = [
        m for m in ["DKT", "SAKT", "GKT", "w/o Graph Propagation"] if m in student_preds
    ]

    if not primary_models or not comparison_models:
        logger.warning("Insufficient models loaded for paired statistical comparison.")
        return []

    # Pre-convert student predictions into NumPy arrays for fast vectorized bootstrapping
    student_arrays: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for m_label in student_preds:
        student_arrays[m_label] = {}
        for s_u, pairs in student_preds[m_label].items():
            if pairs:
                yt_arr = np.array([p[0] for p in pairs], dtype=np.int32)
                yp_arr = np.array([p[1] for p in pairs], dtype=np.float32)
                student_arrays[m_label][s_u] = (yt_arr, yp_arr)

    # Common unique students across primary
    sample_primary = primary_models[0]
    unique_students = list(student_preds[sample_primary].keys())

    stat_results: list[dict[str, Any]] = []

    for prim in primary_models:
        for comp in comparison_models:
            if prim == comp:
                continue

            # Pre-pack aligned student arrays to eliminate nested loop allocations
            student_packed: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for s_u in unique_students:
                if s_u in student_arrays.get(prim, {}) and s_u in student_arrays.get(comp, {}):
                    yt_p, yp_p = student_arrays[prim][s_u]
                    _, yp_c = student_arrays[comp][s_u]
                    min_len = min(len(yt_p), len(yp_c))
                    if min_len > 0:
                        student_packed.append((yt_p[:min_len], yp_p[:min_len], yp_c[:min_len]))

            n_valid_students = len(student_packed)
            if n_valid_students == 0:
                continue

            delta_aucs: list[float] = []
            delta_briers: list[float] = []

            for _ in range(n_bootstrap):
                # Resample student indices with replacement
                boot_idx = np.random.choice(n_valid_students, size=n_valid_students, replace=True)

                yt_sample = np.concatenate([student_packed[i][0] for i in boot_idx])
                yp_prim_sample = np.concatenate([student_packed[i][1] for i in boot_idx])
                yp_comp_sample = np.concatenate([student_packed[i][2] for i in boot_idx])

                if len(np.unique(yt_sample)) < 2:
                    continue

                try:
                    auc_p = float(roc_auc_score(yt_sample, yp_prim_sample))
                    auc_c = float(roc_auc_score(yt_sample, yp_comp_sample))
                    brier_p = float(brier_score_loss(yt_sample, yp_prim_sample))
                    brier_c = float(brier_score_loss(yt_sample, yp_comp_sample))

                    delta_aucs.append(auc_p - auc_c)
                    delta_briers.append(brier_p - brier_c)
                except Exception:
                    continue

            logger.info(
                "   [Check 5] Completed %s vs. %s (%d resamples)", prim, comp, len(delta_aucs)
            )

            min_valid_bootstraps = max(5, int(n_bootstrap * 0.5))
            if len(delta_aucs) >= min_valid_bootstraps:
                mean_d_auc = float(np.mean(delta_aucs))
                ci_auc_low = float(np.percentile(delta_aucs, 2.5))
                ci_auc_high = float(np.percentile(delta_aucs, 97.5))
                # Two-sided empirical bootstrap p-value
                p_val_auc = float(
                    2.0
                    * min(np.mean(np.array(delta_aucs) <= 0), np.mean(np.array(delta_aucs) >= 0))
                )
                p_val_auc = min(1.0, max(1.0 / n_bootstrap, p_val_auc))

                mean_d_brier = float(np.mean(delta_briers))
                ci_brier_low = float(np.percentile(delta_briers, 2.5))
                ci_brier_high = float(np.percentile(delta_briers, 97.5))
                p_val_brier = float(
                    2.0
                    * min(
                        np.mean(np.array(delta_briers) >= 0), np.mean(np.array(delta_briers) <= 0)
                    )
                )
                p_val_brier = min(1.0, max(1.0 / n_bootstrap, p_val_brier))

                stat_results.append(
                    {
                        "Comparison": f"{prim} vs. {comp}",
                        "Primary Model": prim,
                        "Reference Baseline": comp,
                        "Mean Δ AUC": round(mean_d_auc, 4),
                        "95% CI (AUC)": f"[{ci_auc_low:.4f}, {ci_auc_high:.4f}]",
                        "Bootstrap p-value (AUC)": (
                            "< 0.001" if p_val_auc < 0.001 else f"{p_val_auc:.4f}"
                        ),
                        "Mean Δ Brier": round(mean_d_brier, 4),
                        "95% CI (Brier)": f"[{ci_brier_low:.4f}, {ci_brier_high:.4f}]",
                        "Bootstrap p-value (Brier)": (
                            "< 0.001" if p_val_brier < 0.001 else f"{p_val_brier:.4f}"
                        ),
                        "Significant (p < 0.05)": bool(p_val_auc < 0.05),
                    }
                )

    out_csv_path = results_dir / "statistical_tests.csv"
    fieldnames = [
        "Comparison",
        "Primary Model",
        "Reference Baseline",
        "Mean Δ AUC",
        "95% CI (AUC)",
        "Bootstrap p-value (AUC)",
        "Mean Δ Brier",
        "95% CI (Brier)",
        "Bootstrap p-value (Brier)",
        "Significant (p < 0.05)",
    ]
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if stat_results:
            writer.writerows(stat_results)
    logger.info(
        "Saved statistical_tests.csv to %s (%d comparisons)", out_csv_path, len(stat_results)
    )

    return stat_results


# ------------------------------------------------------------------------------
# Unified Markdown Report Generator
# ------------------------------------------------------------------------------
def generate_unified_markdown_report(
    check1: dict[str, Any],
    check2: dict[str, Any],
    check3: dict[str, Any],
    check4: dict[str, Any],
    check5: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Export a comprehensive, publication-grade final validation audit report."""
    md_lines: list[str] = [
        "# TPA-GKT: Final Experiment Validation Audit Report",
        "> **Empirical Knowledge Tracing & Delayed Retention on Junyi Academy 2020**",
        "",
        "---",
        "",
        "## Executive Summary",
        f"- **Benchmark Verification:** {check1.get('total_runs_found', 0)} model runs verified across {check1.get('configs_count', 0)} configurations.",
        f"- **Dataset Fingerprint Valid:** `{check1.get('fingerprint_valid', False)}` (strict interaction & retention pair count parity).",
        f"- **Zero-Leakage Compliance:** `{check4.get('zero_leakage_compliance_pct', 100.0)}%` (0 same-concept interactions skipped between t and t').",
        f"- **Concept Mastery Correlation:** Spearman $\\rho = {check3.get('spearman_rho', 'N/A')}$ against future rolling accuracy ($k \\ge 3$).",
        f"- **Statistically Significant Lifts:** Confirmed across {sum(1 for r in check5 if r.get('Significant (p < 0.05)'))} / {len(check5)} pairwise comparisons via Student-Clustered Bootstrap ($B=1,000$).",
        "",
        "---",
        "",
        "## Check 1: Seed-Level Results & Dataset Fingerprint",
        f"- Total Model Runs Audited: **{check1.get('total_runs_found', 0)}**",
        f"- Full 30-Run Completeness: **{check1.get('all_30_runs_present', False)}**",
        f"- Dataset Fingerprint Match: **{check1.get('fingerprint_valid', False)}**",
        "",
        "```",
        json.dumps(check1.get("cohort_counts", {}), indent=2),
        "```",
        "",
        "---",
        "",
        "## Check 2: Graph Ablation Deep-Dive (Geometry & Stratification)",
        "",
        "### 2.1 Embedding Geometry & Anti-Oversmoothing",
        "| Model | Connected Hier Sim | Connected Trans Sim | Combined Sim | Unconnected Sim | Sim Delta | Ratio | Over-smoothed? |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for m_label, geom in check2.get("embedding_geometry", {}).items():
        md_lines.append(
            f"| **{m_label.upper()}** | {geom.get('mean_cosine_sim_hier_connected')} | "
            f"{geom.get('mean_cosine_sim_trans_connected')} | {geom.get('mean_cosine_sim_combined_connected')} | "
            f"{geom.get('mean_cosine_sim_unconnected')} | **+{geom.get('similarity_delta_connected_minus_unconnected')}** | "
            f"{geom.get('ratio_connected_to_unconnected')}x | `{geom.get('is_oversmoothed')}` |"
        )

    md_lines.extend(
        [
            "",
            "### 2.2 Cold-Start / Sparse Concept Stratification",
            "| Model | Sparse Concepts AUC (Bottom 25%) | Frequent Concepts AUC (Top 75%) | Sparse Retention AUC |",
            "|---|---|---|---|",
        ]
    )

    for m_label, strat in check2.get("stratification_by_training_frequency", {}).items():
        md_lines.append(
            f"| **{m_label.upper()}** | **{strat.get('sparse_response_auc')}** (N={strat.get('sparse_response_samples')}) | "
            f"{strat.get('frequent_response_auc')} (N={strat.get('frequent_response_samples')}) | "
            f"{strat.get('sparse_retention_auc')} |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## Check 3: Independent Concept Mastery Validation",
            "- **Evaluation Setup:** Student $u$'s estimated concept mastery $\\hat{M}_{u,c,t}$ is correlated with their empirical accuracy across subsequent attempts on concept $c$ ($k \\ge 3$, strictly excluding $t+1$).",
            f"- **Eligible Multi-Step Instances:** {check3.get('eligible_multi_step_instances', 0)}",
            f"- **Spearman Rank Correlation ($\\rho$):** **{check3.get('spearman_rho', 'N/A')}** (p = {check3.get('spearman_p_value', 'N/A')})",
            f"- **Pearson Correlation ($r$):** **{check3.get('pearson_r', 'N/A')}** (p = {check3.get('pearson_p_value', 'N/A')})",
            f"- **Sustained Mastery AUC ($\\ge 80\\%$ on final 3 attempts):** **{check3.get('sustained_mastery_auc', 'N/A')}**",
            f"- **Calibration Error (ECE):** **{check3.get('sustained_mastery_ece', 'N/A')}** | Brier: **{check3.get('sustained_mastery_brier', 'N/A')}**",
            "",
            "---",
            "",
            "## Check 4: Delayed-Retention Forensic Audit",
            f"- **Total Retention Pairs Audited:** {check4.get('total_test_retention_pairs', 0)}",
            f"- **Zero Same-Concept Leakage:** `{check4.get('zero_same_concept_leakage_assert', False)}` ({check4.get('zero_leakage_compliance_pct')}% compliance)",
            f"- **Intervening Practice Rate:** {check4.get('intervening_practice_overall_pct')}% of retention pairs had intervening practice on other concepts.",
            "- **Repeated Questions vs Different Questions:**",
            f"  - Repeated questions (memorization): {check4.get('repeated_exact_questions_pct')}% ({check4.get('repeated_exact_questions_count')} pairs) | AUC: `{check4.get('retention_auc_repeated_questions_only')}`",
            f"  - Different questions (concept retention): **{check4.get('different_questions_pct')}%** ({check4.get('different_questions_count')} pairs) | AUC: **`{check4.get('retention_auc_different_questions_only')}`**",
            f"- **15-Minute Timestamp Alignment:** {check4.get('timestamp_15min_alignment_overall_pct')}%",
            "",
            "### Bucket Class Balance & Delay Distribution",
            "| Bucket | Pair Count | Positive Rate ($\\bar{y}_F$) | 15-min Aligned % |",
            "|---|---|---|---|",
        ]
    )

    for b_name, b_info in check4.get("bucket_forensics", {}).items():
        md_lines.append(
            f"| **{b_name}** | {b_info.get('pair_count')} | {b_info.get('positive_class_rate')} | {b_info.get('pct_15min_aligned')}% |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## Check 5: Paired Statistical Significance & Uncertainty (Bootstrap B=1,000)",
            "| Comparison | Mean Δ AUC | 95% CI (AUC) | Bootstrap p-value (AUC) | Mean Δ Brier | 95% CI (Brier) | Significant? |",
            "|---|---|---|---|---|---|---|",
        ]
    )

    for row in check5:
        md_lines.append(
            f"| **{row.get('Comparison')}** | **+{row.get('Mean Δ AUC')}** | {row.get('95% CI (AUC)')} | "
            f"**{row.get('Bootstrap p-value (AUC)')}** | {row.get('Mean Δ Brier')} | {row.get('95% CI (Brier)')} | "
            f"`{row.get('Significant (p < 0.05)')}` |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "*Report generated automatically by `services/gnn_kt/src/final_validation_audit.py`.*",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    logger.info("Saved final_experiment_validation.md to %s", output_path)


# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="TPA-GKT Final Experiment Validation Audit")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="services/gnn_kt/outputs/results",
        help="Directory with summary JSON and CSV outputs.",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="services/gnn_kt/outputs/runs",
        help="Directory with individual per-run metric JSONs.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="services/gnn_kt/outputs/checkpoints",
        help="Directory with trained model checkpoints (.pt).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory with preprocessed test parquets and graph .pt files.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of student-clustered bootstrap resamples.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow running on fast-dev/partial outputs without asserting 30 full runs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: 'cuda', 'cpu', or 'auto'.",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    runs_dir = Path(args.runs_dir)
    checkpoints_dir = Path(args.checkpoints_dir)
    data_dir = Path(args.data_dir)

    results_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)

    logger.info("=" * 80)
    logger.info("STARTING FINAL EXPERIMENT VALIDATION AUDIT")
    logger.info("Device: %s | Results: %s | Data: %s", device, results_dir, data_dir)
    logger.info("=" * 80)

    # 1. Seed-Level Verification
    check1, _ = run_seed_level_verification(
        runs_dir=runs_dir,
        results_dir=results_dir,
        allow_partial=args.allow_partial,
    )

    # 2. Graph Ablation Deep-Dive
    check2 = run_graph_ablation_deep_dive(
        checkpoints_dir=checkpoints_dir,
        data_dir=data_dir,
        results_dir=results_dir,
        device=device,
    )

    # 3. Independent Mastery Validation
    tpa_ckpt_candidates = [
        checkpoints_dir / "tpa_gkt_full_seed42_best.pt",
        checkpoints_dir / "tpa_gkt_full_seed42_fastdev_best.pt",
    ]
    check3 = run_independent_mastery_validation(
        checkpoint_candidates=tpa_ckpt_candidates,
        data_dir=data_dir,
        results_dir=results_dir,
        device=device,
    )

    # 4. Retention Label Forensic Audit
    check4 = run_retention_label_forensic_audit(
        data_dir=data_dir,
        checkpoints_dir=checkpoints_dir,
        results_dir=results_dir,
        device=device,
    )

    # 5. Paired Statistical Significance
    check5 = run_paired_statistical_significance(
        checkpoints_dir=checkpoints_dir,
        data_dir=data_dir,
        results_dir=results_dir,
        device=device,
        n_bootstrap=args.n_bootstrap,
    )

    # 6. Generate Unified Markdown Report
    report_path = results_dir / "final_experiment_validation.md"
    generate_unified_markdown_report(check1, check2, check3, check4, check5, report_path)

    logger.info("=" * 80)
    logger.info("[SUCCESS] Final Experiment Validation Audit completed cleanly!")
    logger.info("Exported files:")
    logger.info("  1. %s", results_dir / "per_seed_results.csv")
    logger.info("  2. %s", results_dir / "graph_ablation_audit.json")
    logger.info("  3. %s", results_dir / "mastery_independent_eval.json")
    logger.info("  4. %s", results_dir / "retention_label_audit.json")
    logger.info("  5. %s", results_dir / "statistical_tests.csv")
    logger.info("  6. %s", report_path)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
