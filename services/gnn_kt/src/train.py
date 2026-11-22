"""
Temporal Prerequisite-Aware Graph Knowledge Tracing (TPA-GKT)
Training Pipeline (train.py)

Trains TPA-GKT and baseline models (DKT, SAKT, GKT) with:
  - Strict device-agnostic execution (CUDA if available, else CPU).
  - Deterministic seed initialization across random, numpy, and torch.
  - AdamW optimizer with gradient clipping.
  - Validation ROC-AUC tracking, model checkpointing, and early stopping.
  - CLI flags and metrics export to outputs/runs/<run_name>_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
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
import yaml
from services.gnn_kt.src.baselines import DKT, GKT, SAKT
from services.gnn_kt.src.dataset import create_dataloader
from services.gnn_kt.src.model import TPAGKT
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch_geometric.data import Data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Set deterministic seeds across all random number generators."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return torch.device for CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vocab_sizes(processed_dir: Path) -> tuple[int, int]:
    """Load concept and problem vocabulary counts from metadata.json or vocabulary files."""
    meta_path = processed_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
            num_concepts = int(meta.get("num_concepts", 150))
            num_problems = int(meta.get("num_problems", 3000))
            logger.info(
                "Loaded vocab sizes from %s: num_concepts=%d, num_problems=%d",
                meta_path,
                num_concepts,
                num_problems,
            )
            return num_concepts, num_problems

    concept_vocab_path = processed_dir / "vocab_concept.json"
    problem_vocab_path = processed_dir / "vocab_problem.json"

    num_concepts = 150
    num_problems = 3000

    if concept_vocab_path.exists():
        with open(concept_vocab_path, encoding="utf-8") as f:
            c_vocab = json.load(f)
            num_concepts = len(c_vocab)

    if problem_vocab_path.exists():
        with open(problem_vocab_path, encoding="utf-8") as f:
            p_vocab = json.load(f)
            num_problems = len(p_vocab)

    logger.info(
        "Loaded vocab sizes from individual json: num_concepts=%d, num_problems=%d",
        num_concepts,
        num_problems,
    )
    return num_concepts, num_problems


def load_graph_tensors(
    processed_dir: Path,
    graph_type: str = "combined",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load preprocessed graph topology edge_index and edge_weight."""
    filename_map = {
        "combined": "A_combined.pt",
        "hier_only": "A_hier.pt",
        "trans_only": "A_trans.pt",
    }
    graph_file = processed_dir / filename_map.get(graph_type, "A_combined.pt")
    if not graph_file.exists():
        logger.warning("Graph file %s not found. Using empty graph tensors.", graph_file)
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float32)

    data: Data = torch.load(graph_file, weights_only=False)
    edge_index = data.edge_index
    if hasattr(data, "edge_weight") and data.edge_weight is not None:
        edge_weight = data.edge_weight
    else:
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)

    return edge_index, edge_weight


def build_model(
    model_name: str,
    num_concepts: int,
    num_problems: int,
    config: dict[str, Any],
    processed_dir: Path,
    device: torch.device,
) -> nn.Module:
    """Instantiate requested model architecture with appropriate config."""
    model_name_lower = model_name.lower().replace("-", "_")

    embedding_dim = int(config.get("embedding_dim", 64))
    hidden_dim = int(config.get("hidden_dim", 128))
    dropout = float(config.get("dropout", 0.2))
    max_seq_len = int(config.get("max_seq_len", 200))

    if model_name_lower == "dkt":
        model = DKT(
            num_concepts=num_concepts,
            num_problems=num_problems,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    elif model_name_lower == "sakt":
        model = SAKT(
            num_concepts=num_concepts,
            num_problems=num_problems,
            embedding_dim=embedding_dim,
            num_heads=int(config.get("gat_heads", 4)),
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
    elif model_name_lower == "gkt":
        edge_index, edge_weight = load_graph_tensors(processed_dir, graph_type="combined")
        model = GKT(
            num_concepts=num_concepts,
            num_problems=num_problems,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            edge_index=edge_index.to(device),
            edge_weight=edge_weight.to(device),
        )
    else:  # "tpa_gkt" or variants
        graph_type = str(config.get("graph_type", "combined"))
        gnn_type = str(config.get("gnn_type", "gcn"))
        use_graph = bool(config.get("use_graph", True))
        use_time_decay = bool(config.get("use_time_decay", True))
        use_attempt_hint = bool(config.get("use_attempt_hint", True))
        use_retention_head = bool(config.get("use_retention_head", True))

        edge_index, edge_weight = load_graph_tensors(processed_dir, graph_type=graph_type)

        model = TPAGKT(
            num_concepts=num_concepts,
            num_problems=num_problems,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=int(config.get("num_gnn_layers", 2)),
            num_rnn_layers=int(config.get("num_rnn_layers", 1)),
            dropout=dropout,
            gat_heads=int(config.get("gat_heads", 4)),
            loss_weight_mastery=float(config.get("loss_weight_mastery", 0.2)),
            loss_weight_retention=float(config.get("loss_weight_retention", 0.1)),
            use_graph=use_graph,
            graph_type=graph_type,
            gnn_type=gnn_type,
            use_time_decay=use_time_decay,
            use_attempt_hint=use_attempt_hint,
            use_retention_head=use_retention_head,
            edge_index=edge_index.to(device),
            edge_weight=edge_weight.to(device),
        )

    return model.to(device)


def evaluate_val_loss_and_auc(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """Compute validation loss, next-response AUC, and retention AUC."""
    model.eval()
    total_val_loss = 0.0
    num_batches = 0

    y_true_list: list[float] = []
    y_pred_list: list[float] = []
    ret_true_list: list[float] = []
    ret_pred_list: list[float] = []

    with torch.no_grad():
        for batch in val_loader:
            out = model(batch)
            total_val_loss += float(out["loss"].item())
            num_batches += 1

            next_mask = batch["next_mask"].to(device)
            next_correct = batch["next_correct"].to(device)
            pred_response = out["pred_response"]

            if next_mask.any():
                y_true_list.extend(next_correct[next_mask].cpu().numpy().tolist())
                y_pred_list.extend(pred_response[next_mask].cpu().numpy().tolist())

            if "pred_retention" in out and "ret_mask" in batch:
                ret_mask = batch["ret_mask"].to(device)
                ret_target = batch["ret_target"].to(device)
                pred_retention = out["pred_retention"]
                if ret_mask.any():
                    ret_true_list.extend(ret_target[ret_mask].cpu().numpy().tolist())
                    ret_pred_list.extend(pred_retention[ret_mask].cpu().numpy().tolist())

    avg_loss = total_val_loss / max(1, num_batches)

    # Next-response ROC-AUC
    response_auc = 0.5
    if len(y_true_list) > 0 and len(set(y_true_list)) > 1:
        try:
            response_auc = float(roc_auc_score(y_true_list, y_pred_list))
        except Exception:
            response_auc = 0.5

    # Delayed-retention ROC-AUC
    retention_auc = 0.5
    if len(ret_true_list) > 0 and len(set(ret_true_list)) > 1:
        try:
            retention_auc = float(roc_auc_score(ret_true_list, ret_pred_list))
        except Exception:
            retention_auc = 0.5

    return avg_loss, response_auc, retention_auc


def train_model(
    model_name: str = "tpa_gkt",
    seed: int = 42,
    fast_dev_run: bool = False,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    patience: int = 5,
    config_overrides: dict[str, Any] | None = None,
    config_path: str | Path = "services/gnn_kt/config.yaml",
    processed_dir_path: str | Path = "data/processed",
    output_dir_path: str | Path = "services/gnn_kt/outputs",
    run_name: str | None = None,
) -> dict[str, Any]:
    """Train a model and return checkpoint path, best epoch, and evaluation metrics."""
    seed_everything(seed)
    device = get_device()
    logger.info("Starting training run for %s on device: %s (seed: %d)", model_name, device, seed)

    # Load base config
    cfg_path = Path(config_path)
    base_config: dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            base_config = yaml.safe_load(f) or {}

    # Merge overrides
    config = {**base_config, **(config_overrides or {})}

    # Resolve training parameters
    n_epochs = epochs if epochs is not None else int(config.get("num_epochs", 25))
    b_size = batch_size if batch_size is not None else int(config.get("batch_size", 64))
    learning_rate = lr if lr is not None else float(config.get("learning_rate", 0.001))
    w_decay = weight_decay if weight_decay is not None else float(config.get("weight_decay", 1e-4))
    grad_clip = float(config.get("grad_clip", 1.0))
    max_seq_len = int(config.get("max_seq_len", 200))

    if fast_dev_run:
        if epochs is None:
            n_epochs = 2
        if batch_size is None:
            b_size = 32
        logger.info("Fast dev run enabled: running %d epochs with batch size %d", n_epochs, b_size)

    # Directories
    processed_dir = Path(processed_dir_path)
    out_dir = Path(output_dir_path)
    checkpoint_dir = out_dir / "checkpoints"
    runs_dir = out_dir / "runs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Run identifier
    if run_name is None:
        run_name = f"{model_name}_seed{seed}"
        if fast_dev_run:
            run_name += "_fastdev"

    checkpoint_path = checkpoint_dir / f"{run_name}_best.pt"
    metrics_path = runs_dir / f"{run_name}_metrics.json"

    # Datasets and DataLoaders
    train_parquet = processed_dir / "train.parquet"
    val_parquet = processed_dir / "val.parquet"

    if not train_parquet.exists() or not val_parquet.exists():
        raise FileNotFoundError(
            f"Required datasets not found in {processed_dir}. Run preprocess.py first."
        )

    train_loader = create_dataloader(
        train_parquet,
        batch_size=b_size,
        shuffle=True,
        max_seq_len=max_seq_len,
    )
    val_loader = create_dataloader(
        val_parquet,
        batch_size=b_size,
        shuffle=False,
        max_seq_len=max_seq_len,
    )

    num_concepts, num_problems = load_vocab_sizes(processed_dir)
    model = build_model(model_name, num_concepts, num_problems, config, processed_dir, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=w_decay)

    best_val_auc = -1.0
    best_epoch = -1
    best_val_ret_auc = -1.0
    patience_counter = 0

    history: list[dict[str, Any]] = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_train_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = out["loss"]

            if torch.isnan(loss) or torch.isinf(loss):
                logger.error("Encountered NaN/Inf loss in epoch %d. Aborting.", epoch)
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            epoch_loss += float(loss.item())
            num_train_batches += 1

        avg_train_loss = epoch_loss / max(1, num_train_batches)
        val_loss, val_auc, val_ret_auc = evaluate_val_loss_and_auc(model, val_loader, device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_retention_auc": val_ret_auc,
        }
        history.append(epoch_record)

        logger.info(
            "Epoch %02d/%02d | TrLoss: %.4f | ValLoss: %.4f | ValAUC: %.4f | RetAUC: %.4f",
            epoch,
            n_epochs,
            avg_train_loss,
            val_loss,
            val_auc,
            val_ret_auc,
        )

        # Checkpoint based on Validation ROC-AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_val_ret_auc = val_ret_auc
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "best_val_auc": best_val_auc,
                    "best_val_ret_auc": best_val_ret_auc,
                    "config": config,
                    "num_concepts": num_concepts,
                    "num_problems": num_problems,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience and not fast_dev_run:
                logger.info("Early stopping triggered at epoch %d (patience=%d).", epoch, patience)
                break

    # Save run summary metrics
    summary = {
        "run_name": run_name,
        "model_name": model_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "best_val_retention_auc": best_val_ret_auc,
        "checkpoint_path": str(checkpoint_path),
        "history": history,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Training complete. Best checkpoint saved to %s", checkpoint_path)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for standalone training."""
    parser = argparse.ArgumentParser(description="TPA-GKT Training Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="tpa_gkt",
        choices=["tpa_gkt", "dkt", "sakt", "gkt"],
        help="Model architecture to train",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fast-dev-run", action="store_true", help="2-epoch sanity check")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--run-name", type=str, default=None, help="Custom run name")

    # Ablation overrides
    parser.add_argument("--no-graph", action="store_true", help="Disable GNN graph propagation")
    parser.add_argument(
        "--graph-type",
        type=str,
        default="combined",
        choices=["combined", "hier_only", "trans_only"],
        help="Graph topology type",
    )
    parser.add_argument(
        "--gnn-type",
        type=str,
        default="gcn",
        choices=["gcn", "gat"],
        help="GNN layer type",
    )
    parser.add_argument("--no-time-decay", action="store_true", help="Disable delta_t time decay")
    parser.add_argument(
        "--no-attempt-hint", action="store_true", help="Disable attempt and hint features"
    )
    parser.add_argument(
        "--no-retention-head", action="store_true", help="Disable delayed retention head"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    overrides: dict[str, Any] = {
        "graph_type": args.graph_type,
        "gnn_type": args.gnn_type,
    }
    if args.no_graph:
        overrides["use_graph"] = False
    if args.no_time_decay:
        overrides["use_time_decay"] = False
    if args.no_attempt_hint:
        overrides["use_attempt_hint"] = False
    if args.no_retention_head:
        overrides["use_retention_head"] = False

    train_model(
        model_name=args.model,
        seed=args.seed,
        fast_dev_run=args.fast_dev_run,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        config_overrides=overrides,
        run_name=args.run_name,
    )
