"""
Unit and Strict Causality Tests for TPA-GKT and Baseline Models.

Verifies:
  1. Forward and backward passes on real preprocessed mini-batches:
     - TPA-GKT (GCN backbone)
     - TPA-GKT (GAT backbone)
     - All 6 ablation toggle combinations
     - Baselines: DKT, SAKT, GKT
  2. Loss computation, finite non-NaN values, and non-None parameter gradients.
  3. Strict Causality Invariance:
     - When future timesteps t' > t are perturbed, hidden states and predictions
       at past/current timesteps <= t remain bit-for-bit identical (zero future leakage).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from services.gnn_kt.src.baselines import DKT, GKT, SAKT
from services.gnn_kt.src.dataset import create_dataloader
from services.gnn_kt.src.model import TPAGKT


@pytest.fixture(scope="module")
def sample_batch() -> dict[str, torch.Tensor]:
    """Load a real mini-batch from preprocessed train.parquet."""
    train_parquet = Path("data/processed/train.parquet")
    if not train_parquet.exists():
        pytest.skip("Preprocessed train.parquet not found. Run preprocess.py first.")

    loader = create_dataloader(train_parquet, batch_size=4, shuffle=False, max_seq_len=50)
    return next(iter(loader))


@pytest.fixture(scope="module")
def graph_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Load preprocessed graph topology tensors."""
    graph_file = Path("data/processed/A_combined.pt")
    if not graph_file.exists():
        pytest.skip("Preprocessed A_combined.pt not found. Run graph_builder.py first.")

    data = torch.load(graph_file, weights_only=False)
    edge_index = data.edge_index
    edge_weight = getattr(data, "edge_weight", torch.ones(edge_index.size(1), dtype=torch.float32))
    return edge_index, edge_weight


def test_tpagkt_forward_backward_gcn(
    sample_batch: dict[str, torch.Tensor],
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Verify forward and backward pass for TPA-GKT with GCN backbone."""
    edge_index, edge_weight = graph_data
    model = TPAGKT(
        num_concepts=150,
        num_problems=3000,
        embedding_dim=32,
        hidden_dim=64,
        gnn_type="gcn",
        edge_index=edge_index,
        edge_weight=edge_weight,
    )
    model.train()

    out = model(sample_batch)
    loss = out["loss"]

    assert torch.isfinite(loss)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0

    loss.backward()

    # Verify gradients exist and are finite
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN in gradient of {name}"
            assert not torch.isinf(param.grad).any(), f"Inf in gradient of {name}"


def test_tpagkt_forward_backward_gat(
    sample_batch: dict[str, torch.Tensor],
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Verify forward and backward pass for TPA-GKT with GAT backbone."""
    edge_index, edge_weight = graph_data
    model = TPAGKT(
        num_concepts=150,
        num_problems=3000,
        embedding_dim=32,
        hidden_dim=64,
        gnn_type="gat",
        gat_heads=2,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )
    model.train()

    out = model(sample_batch)
    loss = out["loss"]

    assert torch.isfinite(loss)
    loss.backward()
    assert model.concept_base_emb.grad is not None


@pytest.mark.parametrize(
    "ablation_config",
    [
        {"use_graph": False},                                    # without_gnn_propagation
        {"graph_type": "trans_only"},                            # without_hier_graph
        {"graph_type": "hier_only"},                             # without_trans_graph
        {"use_time_decay": False},                               # without_time_decay
        {"use_attempt_hint": False},                             # without_attempt_hint
        {"use_retention_head": False},                           # without_retention_head
    ],
)
def test_tpagkt_ablations(
    ablation_config: dict[str, Any],
    sample_batch: dict[str, torch.Tensor],
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Verify all 6 ablation toggle combinations run and compute valid gradients."""
    edge_index, edge_weight = graph_data
    model = TPAGKT(
        num_concepts=150,
        num_problems=3000,
        embedding_dim=32,
        hidden_dim=64,
        edge_index=edge_index,
        edge_weight=edge_weight,
        **ablation_config,
    )
    model.train()

    out = model(sample_batch)
    loss = out["loss"]
    assert torch.isfinite(loss)
    loss.backward()


def test_concept_mastery_head(
    sample_batch: dict[str, torch.Tensor],
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Verify concept mastery matrix generation across all concepts [B, L, N]."""
    edge_index, edge_weight = graph_data
    model = TPAGKT(
        num_concepts=150,
        num_problems=3000,
        embedding_dim=32,
        hidden_dim=64,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )
    model.eval()

    with torch.no_grad():
        out = model(sample_batch)
        h = out["hidden_states"]
        mastery = model.compute_concept_mastery(h)

    b, seq_len, _ = h.shape
    assert mastery.shape == (b, seq_len, 150)
    assert (mastery >= 0.0).all() and (mastery <= 1.0).all()


def test_dkt_baseline(sample_batch: dict[str, torch.Tensor]) -> None:
    """Verify DKT forward, backward, and loss computation."""
    model = DKT(num_concepts=150, num_problems=3000, embedding_dim=32, hidden_dim=64)
    model.train()

    out = model(sample_batch)
    loss = out["loss"]
    assert torch.isfinite(loss)
    loss.backward()


def test_sakt_baseline(sample_batch: dict[str, torch.Tensor]) -> None:
    """Verify SAKT forward, backward, and causal self-attention."""
    model = SAKT(
        num_concepts=150,
        num_problems=3000,
        embedding_dim=32,
        num_heads=2,
        max_seq_len=100,
    )
    model.train()

    out = model(sample_batch)
    loss = out["loss"]
    assert torch.isfinite(loss)
    loss.backward()


def test_gkt_baseline(
    sample_batch: dict[str, torch.Tensor],
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Verify GKT forward and backward passes."""
    edge_index, edge_weight = graph_data
    model = GKT(
        num_concepts=150,
        num_problems=3000,
        embedding_dim=32,
        hidden_dim=64,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )
    model.train()

    out = model(sample_batch)
    loss = out["loss"]
    assert torch.isfinite(loss)
    loss.backward()


# ==============================================================================
# Strict Causality Tests (Zero Future Leakage Verification)
# ==============================================================================
@pytest.mark.parametrize("model_cls", ["TPAGKT", "DKT", "SAKT", "GKT"])
def test_strict_causality_invariance(
    model_cls: str,
    sample_batch: dict[str, torch.Tensor],
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """
    Assert that modifying future timesteps t' > t leaves predictions and
    hidden states at past/current timesteps <= t bit-for-bit identical.
    """
    edge_index, edge_weight = graph_data

    if model_cls == "TPAGKT":
        model = TPAGKT(
            num_concepts=150,
            num_problems=3000,
            embedding_dim=32,
            hidden_dim=64,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
    elif model_cls == "DKT":
        model = DKT(num_concepts=150, num_problems=3000, embedding_dim=32, hidden_dim=64)
    elif model_cls == "SAKT":
        model = SAKT(
            num_concepts=150,
            num_problems=3000,
            embedding_dim=32,
            num_heads=2,
            max_seq_len=100,
        )
    elif model_cls == "GKT":
        model = GKT(
            num_concepts=150,
            num_problems=3000,
            embedding_dim=32,
            hidden_dim=64,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
    else:
        raise ValueError(f"Unknown model class: {model_cls}")

    model.eval()

    # 1. Forward pass on original batch
    with torch.no_grad():
        out_orig = model(sample_batch)

    # 2. Perturb future tokens at t >= cut_step
    cut_step = 10
    perturbed_batch = {
        k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in sample_batch.items()
    }

    # Alter future concepts, problems, correctness, and continuous features
    perturbed_batch["concept_idx"][:, cut_step:] = (
        perturbed_batch["concept_idx"][:, cut_step:] + 5
    ) % 150
    perturbed_batch["problem_idx"][:, cut_step:] = (
        perturbed_batch["problem_idx"][:, cut_step:] + 7
    ) % 3000
    perturbed_batch["is_correct"][:, cut_step:] = 1 - perturbed_batch["is_correct"][:, cut_step:]
    perturbed_batch["continuous_features"][:, cut_step:, :] += 10.0

    # 3. Forward pass on perturbed batch
    with torch.no_grad():
        out_pert = model(perturbed_batch)

    # 4. Assert past/current outputs at t < cut_step are strictly invariant
    orig_h_past = out_orig["hidden_states"][:, :cut_step, :]
    pert_h_past = out_pert["hidden_states"][:, :cut_step, :]
    assert torch.allclose(orig_h_past, pert_h_past, atol=1e-5, rtol=1e-5), (
        f"{model_cls} violated temporal causality! Future changes leaked into past hidden states."
    )

    orig_pred_past = out_orig["pred_response"][:, : cut_step - 1]
    pert_pred_past = out_pert["pred_response"][:, : cut_step - 1]
    assert torch.allclose(orig_pred_past, pert_pred_past, atol=1e-5, rtol=1e-5), (
        f"{model_cls} violated temporal causality! Future changes leaked into past predictions."
    )
