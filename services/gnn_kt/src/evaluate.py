"""
Temporal Prerequisite-Aware Graph Knowledge Tracing (TPA-GKT)
Paper-Grade Evaluator (evaluate.py)

Evaluates trained model checkpoints on validation or test sets with:
  1. Next-Response Metrics: ROC-AUC, ACC, Precision, Recall, F1, Log Loss, Brier Score.
  2. Delayed-Retention Metrics: Overall ROC-AUC, F1, Brier Score + Per-Bucket
     metrics across [<1h, 1h-24h, 1d-7d, >7d].
  3. Mastery Calibration: Expected Calibration Error (ECE, 10 bins) and reliability diagram bins.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to sys.path for direct CLI script execution
_project_root = str(Path(__file__).resolve().parents[3])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
import torch.nn as nn
from services.gnn_kt.src.dataset import create_dataloader
from services.gnn_kt.src.train import build_model, get_device
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, list[dict[str, Any]]]:
    """
    Compute Expected Calibration Error (ECE) and reliability diagram bin statistics.

    Args:
      y_true: Binary ground truth labels in {0, 1}.
      y_pred: Predicted probability estimates in [0, 1].
      n_bins: Number of equal-width calibration bins (default: 10).

    Returns:
      ece: Scalar expected calibration error.
      bin_records: List of per-bin dictionary records (acc, conf, count, bounds).
    """
    if len(y_true) == 0:
        return 0.0, []

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    total_samples = len(y_pred)
    ece = 0.0
    bin_records: list[dict[str, Any]] = []

    for i in range(n_bins):
        bin_lower = float(bin_boundaries[i])
        bin_upper = float(bin_boundaries[i + 1])

        if i == n_bins - 1:
            in_bin = (y_pred >= bin_lower) & (y_pred <= bin_upper)
        else:
            in_bin = (y_pred >= bin_lower) & (y_pred < bin_upper)

        bin_count = int(np.sum(in_bin))

        if bin_count > 0:
            bin_acc = float(np.mean(y_true[in_bin]))
            bin_conf = float(np.mean(y_pred[in_bin]))
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)
        else:
            bin_acc = 0.0
            bin_conf = 0.0

        bin_records.append(
            {
                "bin_index": i,
                "bin_lower": round(bin_lower, 4),
                "bin_upper": round(bin_upper, 4),
                "count": bin_count,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
            }
        )

    return float(ece), bin_records


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> dict[str, float | None]:
    """Compute binary classification metrics with zero-division protection."""
    if len(y_true) == 0:
        return {
            f"{prefix}auc": None,
            f"{prefix}acc": None,
            f"{prefix}precision": None,
            f"{prefix}recall": None,
            f"{prefix}f1": None,
            f"{prefix}log_loss": None,
            f"{prefix}brier": None,
            f"{prefix}count": 0,
        }

    # AUC calculation with single-class protection
    unique_classes = np.unique(y_true)
    if len(unique_classes) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_pred))
        except Exception:
            auc = 0.5
    else:
        auc = 0.5

    binary_pred = (y_pred >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, binary_pred))
    prec = float(precision_score(y_true, binary_pred, zero_division=0))
    rec = float(recall_score(y_true, binary_pred, zero_division=0))
    f1 = float(f1_score(y_true, binary_pred, zero_division=0))

    clipped_pred = np.clip(y_pred, 1e-7, 1.0 - 1e-7)
    try:
        ll = float(log_loss(y_true, clipped_pred))
    except Exception:
        ll = 0.0

    brier = float(brier_score_loss(y_true, y_pred))

    return {
        f"{prefix}auc": auc,
        f"{prefix}acc": acc,
        f"{prefix}precision": prec,
        f"{prefix}recall": rec,
        f"{prefix}f1": f1,
        f"{prefix}log_loss": ll,
        f"{prefix}brier": brier,
        f"{prefix}count": len(y_true),
    }


def evaluate_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Run evaluation over DataLoader and compute full paper-grade metric suite."""
    model.eval()

    y_resp_true: list[int] = []
    y_resp_pred: list[float] = []

    y_ret_true: list[int] = []
    y_ret_pred: list[float] = []
    y_ret_bucket: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            out = model(batch)

            next_mask = batch["next_mask"].to(device)
            next_correct = batch["next_correct"].to(device)
            pred_response = out["pred_response"]

            if next_mask.any():
                y_resp_true.extend(next_correct[next_mask].cpu().numpy().astype(int).tolist())
                y_resp_pred.extend(pred_response[next_mask].cpu().numpy().astype(float).tolist())

            if "pred_retention" in out and "ret_mask" in batch:
                ret_mask = batch["ret_mask"].to(device)
                ret_target = batch["ret_target"].to(device)
                ret_bucket = batch["ret_bucket"].to(device)
                pred_retention = out["pred_retention"]

                if ret_mask.any():
                    y_ret_true.extend(ret_target[ret_mask].cpu().numpy().astype(int).tolist())
                    y_ret_pred.extend(
                        pred_retention[ret_mask].cpu().numpy().astype(float).tolist()
                    )
                    y_ret_bucket.extend(
                        ret_bucket[ret_mask].cpu().numpy().astype(int).tolist()
                    )

    resp_true_arr = np.array(y_resp_true, dtype=int)
    resp_pred_arr = np.array(y_resp_pred, dtype=float)

    # 1. Next-Response Metrics
    metrics: dict[str, Any] = compute_binary_metrics(
        resp_true_arr, resp_pred_arr, prefix="response_"
    )
    ece_score, reliability_bins = compute_ece(resp_true_arr, resp_pred_arr, n_bins=10)
    metrics["response_ece"] = ece_score
    metrics["reliability_diagram"] = reliability_bins

    # 2. Delayed-Retention Metrics (Overall & Bucketed)
    if len(y_ret_true) > 0:
        ret_true_arr = np.array(y_ret_true, dtype=int)
        ret_pred_arr = np.array(y_ret_pred, dtype=float)
        ret_bucket_arr = np.array(y_ret_bucket, dtype=int)

        ret_overall = compute_binary_metrics(ret_true_arr, ret_pred_arr, prefix="retention_")
        metrics.update(ret_overall)

        bucket_names = {
            0: "<1h",
            1: "1h-24h",
            2: "1d-7d",
            3: ">7d",
        }
        ret_bucket_metrics: dict[str, Any] = {}
        for b_idx, b_name in bucket_names.items():
            b_mask = ret_bucket_arr == b_idx
            b_true = ret_true_arr[b_mask]
            b_pred = ret_pred_arr[b_mask]
            b_res = compute_binary_metrics(b_true, b_pred, prefix="")
            ret_bucket_metrics[b_name] = {
                "bucket_idx": b_idx,
                "name": b_name,
                "auc": b_res["auc"],
                "f1": b_res["f1"],
                "brier": b_res["brier"],
                "count": b_res["count"],
            }
        metrics["retention_buckets"] = ret_bucket_metrics
    else:
        metrics["retention_auc"] = None
        metrics["retention_f1"] = None
        metrics["retention_brier"] = None
        metrics["retention_buckets"] = {}

    return metrics


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    data_parquet_path: str | Path = "data/processed/test.parquet",
    processed_dir_path: str | Path = "data/processed",
    output_json_path: str | Path | None = None,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Load model checkpoint, instantiate architecture, and evaluate on dataset."""
    device = get_device()
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_name = checkpoint.get("model_name", "tpa_gkt")
    config = checkpoint.get("config", {})
    num_concepts = checkpoint.get("num_concepts", 150)
    num_problems = checkpoint.get("num_problems", 3000)

    processed_dir = Path(processed_dir_path)
    model = build_model(model_name, num_concepts, num_problems, config, processed_dir, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_parquet = Path(data_parquet_path)
    if not test_parquet.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at: {test_parquet}")

    dataloader = create_dataloader(
        test_parquet,
        batch_size=batch_size,
        shuffle=False,
        max_seq_len=int(config.get("max_seq_len", 200)),
    )

    metrics = evaluate_dataloader(model, dataloader, device)
    metrics["model_name"] = model_name
    metrics["checkpoint"] = str(ckpt_path)
    metrics["epoch"] = checkpoint.get("epoch", -1)

    if output_json_path is not None:
        out_p = Path(output_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Evaluation metrics exported to %s", out_p)

    return metrics


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for standalone evaluation."""
    parser = argparse.ArgumentParser(description="TPA-GKT Evaluator")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to saved model checkpoint (.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/test.parquet",
        help="Path to evaluation parquet (val.parquet or test.parquet)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to output evaluation JSON",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_parquet_path=args.data,
        output_json_path=args.output_json,
        batch_size=args.batch_size,
    )
    logger.info(
        "Response AUC: %s | Response F1: %s | Retention AUC: %s",
        results.get("response_auc"),
        results.get("response_f1"),
        results.get("retention_auc"),
    )
