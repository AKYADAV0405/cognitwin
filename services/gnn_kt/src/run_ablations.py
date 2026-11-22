"""
Temporal Prerequisite-Aware Graph Knowledge Tracing (TPA-GKT)
Automated Multi-Seed & Ablation Runner (run_ablations.py)

Orchestrates full experimental benchmarking:
  1. Baselines Comparison: DKT, SAKT, GKT vs. TPA-GKT (Full)
  2. All 6 TPA-GKT Ablation Variants:
     - w/o Curriculum Graph (A_hier)
     - w/o Transition Graph (A_trans)
     - w/o Time Decay (Delta t)
     - w/o Attempt & Hint Features
     - w/o Graph Propagation (No GNN)
     - GAT Backbone (vs GCN)
  3. Multi-seed training and test set evaluation across seeds [42, 100, 2026]
     (or single seed [42] in --fast-dev-run).
  4. Exports publication-grade CSV tables to services/gnn_kt/outputs/results/:
     - table3_results.csv
     - ablations.csv
     - retention_buckets.csv
"""

from __future__ import annotations

import argparse
import csv
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
from services.gnn_kt.src.evaluate import evaluate_checkpoint
from services.gnn_kt.src.train import train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Experiment Definitions
# ------------------------------------------------------------------------------
EXPERIMENT_CONFIGS: dict[str, dict[str, Any]] = {
    # Baselines
    "dkt": {
        "model_name": "dkt",
        "label": "DKT",
        "is_baseline": True,
        "is_ablation": False,
        "description": "Deep Knowledge Tracing (Piech et al.)",
        "overrides": {},
    },
    "sakt": {
        "model_name": "sakt",
        "label": "SAKT",
        "is_baseline": True,
        "is_ablation": False,
        "description": "Self-Attentive Knowledge Tracing (Pandey & Karypis)",
        "overrides": {},
    },
    "gkt": {
        "model_name": "gkt",
        "label": "GKT",
        "is_baseline": True,
        "is_ablation": False,
        "description": "Graph Knowledge Tracing (Nakagawa et al.)",
        "overrides": {},
    },
    # Full Model
    "tpa_gkt_full": {
        "model_name": "tpa_gkt",
        "label": "TPA-GKT (Ours)",
        "is_baseline": True,
        "is_ablation": True,
        "description": "Full TPA-GKT (Dual Graph + Time Decay + Multi-Head)",
        "overrides": {
            "use_graph": True,
            "graph_type": "combined",
            "gnn_type": "gcn",
            "use_time_decay": True,
            "use_attempt_hint": True,
            "use_retention_head": True,
        },
    },
    # 6 Ablation Variants
    "ablation_wo_hier": {
        "model_name": "tpa_gkt",
        "label": "w/o Curriculum Graph (A_hier)",
        "is_baseline": False,
        "is_ablation": True,
        "description": "Uses empirical transition graph only (alpha = 0.0)",
        "overrides": {
            "use_graph": True,
            "graph_type": "trans_only",
            "gnn_type": "gcn",
            "use_time_decay": True,
            "use_attempt_hint": True,
            "use_retention_head": True,
        },
    },
    "ablation_wo_trans": {
        "model_name": "tpa_gkt",
        "label": "w/o Transition Graph (A_trans)",
        "is_baseline": False,
        "is_ablation": True,
        "description": "Uses curriculum hierarchy graph only (alpha = 1.0)",
        "overrides": {
            "use_graph": True,
            "graph_type": "hier_only",
            "gnn_type": "gcn",
            "use_time_decay": True,
            "use_attempt_hint": True,
            "use_retention_head": True,
        },
    },
    "ablation_wo_time_decay": {
        "model_name": "tpa_gkt",
        "label": "w/o Time Decay (Delta t)",
        "is_baseline": False,
        "is_ablation": True,
        "description": "Zeroes out temporal elapsed study gap signals",
        "overrides": {
            "use_graph": True,
            "graph_type": "combined",
            "gnn_type": "gcn",
            "use_time_decay": False,
            "use_attempt_hint": True,
            "use_retention_head": True,
        },
    },
    "ablation_wo_attempt_hint": {
        "model_name": "tpa_gkt",
        "label": "w/o Attempt & Hint Features",
        "is_baseline": False,
        "is_ablation": True,
        "description": "Restricts multi-modal features to concept & correctness only",
        "overrides": {
            "use_graph": True,
            "graph_type": "combined",
            "gnn_type": "gcn",
            "use_time_decay": True,
            "use_attempt_hint": False,
            "use_retention_head": True,
        },
    },
    "ablation_wo_graph": {
        "model_name": "tpa_gkt",
        "label": "w/o Graph Propagation (No GNN)",
        "is_baseline": False,
        "is_ablation": True,
        "description": "Bypasses Stage 1 GNN message passing entirely",
        "overrides": {
            "use_graph": False,
            "graph_type": "combined",
            "gnn_type": "gcn",
            "use_time_decay": True,
            "use_attempt_hint": True,
            "use_retention_head": True,
        },
    },
    "ablation_gat": {
        "model_name": "tpa_gkt",
        "label": "GAT Backbone (vs GCN)",
        "is_baseline": False,
        "is_ablation": True,
        "description": "Replaces spectral GCN convolution with multi-head GAT",
        "overrides": {
            "use_graph": True,
            "graph_type": "combined",
            "gnn_type": "gat",
            "use_time_decay": True,
            "use_attempt_hint": True,
            "use_retention_head": True,
        },
    },
}


def format_mean_std(values: list[float | None]) -> str:
    """Format list of numeric values into 'mean ± std' string or 'N/A'."""
    clean_vals = [v for v in values if v is not None and not np.isnan(v)]
    if not clean_vals:
        return "N/A"
    mean_val = float(np.mean(clean_vals))
    if len(clean_vals) == 1:
        return f"{mean_val:.4f}"
    std_val = float(np.std(clean_vals))
    return f"{mean_val:.4f} ± {std_val:.4f}"


def run_experiment_suite(
    seeds: list[int],
    fast_dev_run: bool = False,
    results_dir: str | Path = "services/gnn_kt/outputs/results",
    checkpoints_dir: str | Path = "services/gnn_kt/outputs/checkpoints",
    processed_dir: str | Path = "data/processed",
) -> dict[str, Any]:
    """Execute complete benchmarking across all models, ablations, and seeds."""
    res_path = Path(results_dir)
    ckpt_path = Path(checkpoints_dir)
    proc_path = Path(processed_dir)
    res_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    test_parquet = proc_path / "test.parquet"
    if not test_parquet.exists():
        raise FileNotFoundError(f"test.parquet not found in {proc_path}. Run preprocess.py first.")

    # Results store: {exp_key: {seed: metrics_dict}}
    all_runs: dict[str, dict[int, dict[str, Any]]] = {k: {} for k in EXPERIMENT_CONFIGS}

    total_experiments = len(EXPERIMENT_CONFIGS) * len(seeds)
    current_exp_idx = 0

    logger.info("=" * 70)
    logger.info(
        "Starting Experiment Suite (%d experiments across %d seeds)",
        total_experiments,
        len(seeds),
    )
    logger.info("=" * 70)

    for exp_key, spec in EXPERIMENT_CONFIGS.items():
        for seed in seeds:
            current_exp_idx += 1
            run_name = f"{exp_key}_seed{seed}"
            if fast_dev_run:
                run_name += "_fastdev"

            logger.info(
                "[%d/%d] Running %s (%s) with seed %d...",
                current_exp_idx,
                total_experiments,
                spec["label"],
                exp_key,
                seed,
            )

            # 1. Train model
            train_summary = train_model(
                model_name=spec["model_name"],
                seed=seed,
                fast_dev_run=fast_dev_run,
                config_overrides=spec["overrides"],
                run_name=run_name,
            )

            # 2. Evaluate checkpoint on test.parquet
            saved_checkpoint = train_summary["checkpoint_path"]
            eval_metrics = evaluate_checkpoint(
                checkpoint_path=saved_checkpoint,
                data_parquet_path=test_parquet,
                processed_dir_path=proc_path,
                batch_size=32 if fast_dev_run else 64,
            )

            all_runs[exp_key][seed] = eval_metrics
            logger.info(
                "Finished %s (seed %d) -> Test Response AUC: %s | Test Retention AUC: %s",
                spec["label"],
                seed,
                eval_metrics.get("response_auc"),
                eval_metrics.get("retention_auc"),
            )

    # --------------------------------------------------------------------------
    # Export Table 3: Baselines vs. TPA-GKT (Ours)
    # --------------------------------------------------------------------------
    table3_path = res_path / "table3_results.csv"
    table3_rows: list[dict[str, str]] = []

    baseline_keys = ["dkt", "sakt", "gkt", "tpa_gkt_full"]
    for b_key in baseline_keys:
        spec = EXPERIMENT_CONFIGS[b_key]
        seed_results = list(all_runs[b_key].values())

        resp_aucs = [r.get("response_auc") for r in seed_results]
        resp_accs = [r.get("response_acc") for r in seed_results]
        resp_f1_macros = [r.get("response_f1_macro") for r in seed_results]
        resp_f1_weighteds = [r.get("response_f1_weighted") for r in seed_results]
        resp_f1_bins = [r.get("response_f1") for r in seed_results]
        resp_briers = [r.get("response_brier") for r in seed_results]
        ret_aucs = [r.get("retention_auc") for r in seed_results]
        ret_f1_macros = [r.get("retention_f1_macro") for r in seed_results]

        table3_rows.append(
            {
                "Model": spec["label"],
                "Next-Response AUC": format_mean_std(resp_aucs),
                "Next-Response ACC": format_mean_std(resp_accs),
                "Next-Response Macro-F1": format_mean_std(resp_f1_macros),
                "Next-Response Weighted-F1": format_mean_std(resp_f1_weighteds),
                "Next-Response Binary-F1": format_mean_std(resp_f1_bins),
                "Next-Response Brier": format_mean_std(resp_briers),
                "Retention AUC": format_mean_std(ret_aucs),
                "Retention Macro-F1": format_mean_std(ret_f1_macros),
            }
        )

    with open(table3_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table3_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table3_rows)
    logger.info("Table 3 exported to %s", table3_path)

    # --------------------------------------------------------------------------
    # Export Ablations CSV: Full vs. All 6 Ablations
    # --------------------------------------------------------------------------
    ablations_path = res_path / "ablations.csv"
    ablation_rows: list[dict[str, str]] = []

    ablation_keys = [
        "tpa_gkt_full",
        "ablation_wo_hier",
        "ablation_wo_trans",
        "ablation_wo_time_decay",
        "ablation_wo_attempt_hint",
        "ablation_wo_graph",
        "ablation_gat",
    ]
    for a_key in ablation_keys:
        spec = EXPERIMENT_CONFIGS[a_key]
        seed_results = list(all_runs[a_key].values())

        resp_aucs = [r.get("response_auc") for r in seed_results]
        resp_f1_macros = [r.get("response_f1_macro") for r in seed_results]
        resp_briers = [r.get("response_brier") for r in seed_results]
        ret_aucs = [r.get("retention_auc") for r in seed_results]
        ret_gt7d_aucs = [
            r.get("retention_buckets", {}).get(">7d", {}).get("auc") for r in seed_results
        ]

        ablation_rows.append(
            {
                "Variant": spec["label"],
                "Description": spec["description"],
                "Next-Response AUC": format_mean_std(resp_aucs),
                "Next-Response Macro-F1": format_mean_std(resp_f1_macros),
                "Next-Response Brier": format_mean_std(resp_briers),
                "Overall Retention AUC": format_mean_std(ret_aucs),
                "Retention AUC (>7d)": format_mean_std(ret_gt7d_aucs),
            }
        )

    with open(ablations_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ablation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ablation_rows)
    logger.info("Ablations table exported to %s", ablations_path)

    # --------------------------------------------------------------------------
    # Export Retention Buckets CSV (<1h, 1h-24h, 1d-7d, >7d) from TPA-GKT (Full)
    # --------------------------------------------------------------------------
    retention_buckets_path = res_path / "retention_buckets.csv"
    bucket_rows: list[dict[str, Any]] = []

    full_seed_results = list(all_runs["tpa_gkt_full"].values())
    bucket_names = ["<1h", "1h-24h", "1d-7d", ">7d"]
    bucket_descriptions = {
        "<1h": "Immediate practice (<= 1 hour)",
        "1h-24h": "Same-day review (1 hour to 24 hours)",
        "1d-7d": "Multi-day review (1 day to 7 days)",
        ">7d": "Long-term delay (> 7 days)",
    }

    for b_name in bucket_names:
        b_aucs: list[float | None] = []
        b_f1_macros: list[float | None] = []
        b_f1_bins: list[float | None] = []
        b_briers: list[float | None] = []
        counts: list[int] = []

        for r in full_seed_results:
            b_dict = r.get("retention_buckets", {}).get(b_name, {})
            b_aucs.append(b_dict.get("auc"))
            b_f1_macros.append(b_dict.get("f1_macro"))
            b_f1_bins.append(b_dict.get("f1"))
            b_briers.append(b_dict.get("brier"))
            if "count" in b_dict:
                counts.append(int(b_dict["count"]))

        avg_count = int(np.mean(counts)) if counts else 0
        bucket_rows.append(
            {
                "Bucket": b_name,
                "Time Interval": bucket_descriptions.get(b_name, b_name),
                "TPA-GKT Retention AUC": format_mean_std(b_aucs),
                "Retention Macro-F1": format_mean_std(b_f1_macros),
                "Retention Binary-F1": format_mean_std(b_f1_bins),
                "Retention Brier": format_mean_std(b_briers),
                "Sample Count": avg_count,
            }
        )

    with open(retention_buckets_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(bucket_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bucket_rows)
    logger.info("Retention buckets table exported to %s", retention_buckets_path)

    # Export full summary JSON
    summary_json_path = res_path / "all_results_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2)

    return {
        "all_runs": all_runs,
        "table3_path": str(table3_path),
        "ablations_path": str(ablations_path),
        "retention_buckets_path": str(retention_buckets_path),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for automated runner."""
    parser = argparse.ArgumentParser(description="TPA-GKT Multi-Seed & Ablation Runner")
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Fast sanity check: 2 epochs, 1 seed [42] on CPU",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="List of random seeds (default: [42, 100, 2026], or [42] for fast-dev-run)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.seeds is not None:
        chosen_seeds = args.seeds
    elif args.fast_dev_run:
        chosen_seeds = [42]
    else:
        chosen_seeds = [42, 100, 2026]

    run_experiment_suite(seeds=chosen_seeds, fast_dev_run=args.fast_dev_run)
