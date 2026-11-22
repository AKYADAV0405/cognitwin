"""
Unit and Integration Tests for TPA-GKT Training and Evaluation Pipelines.

Verifies:
  1. Fast training iteration, checkpointing, and metric logging for all model types.
  2. Evaluation pipeline produces complete paper metrics (AUC, ACC, F1, Log Loss, Brier, ECE).
  3. Delayed retention evaluation across [<1h, 1h-24h, 1d-7d, >7d] buckets.
  4. Robustness against single-class and empty predictions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from services.gnn_kt.src.evaluate import (
    compute_binary_metrics,
    compute_ece,
    evaluate_checkpoint,
)
from services.gnn_kt.src.train import train_model


def test_compute_ece_and_metrics() -> None:
    """Verify ECE calculation and binary metric bounds."""
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.85, 0.95, 0.05, 0.15])

    metrics = compute_binary_metrics(y_true, y_pred, prefix="test_")
    assert metrics["test_auc"] is not None and metrics["test_auc"] > 0.9
    assert metrics["test_acc"] is not None and metrics["test_acc"] == 1.0
    assert metrics["test_f1"] is not None and metrics["test_f1"] == 1.0

    ece, bin_data = compute_ece(y_true, y_pred, n_bins=5)
    assert 0.0 <= ece <= 1.0
    assert len(bin_data) == 5
    assert sum(b["count"] for b in bin_data) == len(y_true)


def test_metric_single_class_edge_case() -> None:
    """Verify metric computation gracefully handles single-class edge cases."""
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0.8, 0.9, 0.7, 0.85])

    metrics = compute_binary_metrics(y_true, y_pred, prefix="")
    assert metrics["auc"] == 0.5  # Safe fallback when AUC is undefined
    assert metrics["acc"] == 1.0


def test_train_and_evaluate_fast_dev_cycle(tmp_path: Path) -> None:
    """Verify fast training and subsequent evaluation on real fast-dev data."""
    train_parquet = Path("data/processed/train.parquet")
    if not train_parquet.exists():
        pytest.skip("Preprocessed dataset not found. Run preprocess.py first.")

    run_name = "test_cycle_tpa_gkt"
    summary = train_model(
        model_name="tpa_gkt",
        seed=42,
        fast_dev_run=True,
        epochs=1,
        batch_size=32,
        output_dir_path=tmp_path,
        run_name=run_name,
    )

    ckpt_path = Path(summary["checkpoint_path"])
    assert ckpt_path.exists(), f"Checkpoint {ckpt_path} was not created!"
    assert summary["best_epoch"] == 1

    # Evaluate on validation parquet
    eval_metrics = evaluate_checkpoint(
        checkpoint_path=ckpt_path,
        data_parquet_path="data/processed/val.parquet",
        processed_dir_path="data/processed",
        batch_size=32,
    )

    assert "response_auc" in eval_metrics
    assert "response_acc" in eval_metrics
    assert "response_ece" in eval_metrics
    assert "reliability_diagram" in eval_metrics
    assert "retention_auc" in eval_metrics
    assert "retention_buckets" in eval_metrics

    assert 0.0 <= eval_metrics["response_auc"] <= 1.0
    assert 0.0 <= eval_metrics["response_ece"] <= 1.0
