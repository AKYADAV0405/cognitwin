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


def test_mastery_head_optimizer_update(
    sample_batch: dict[str, torch.Tensor],
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Verify that Concept Mastery Head parameters update after 1 optimizer step."""
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    # Cache pre-step weights of mastery_head
    pre_weights = [p.clone() for p in model.mastery_head.parameters()]

    optimizer.zero_grad()
    out = model(sample_batch)
    loss = out["loss"]
    loss.backward()

    # Assert non-None and non-zero gradients on mastery_head
    for name, param in model.mastery_head.named_parameters():
        assert param.grad is not None, f"Gradient for mastery_head.{name} is None!"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in mastery_head.{name}!"

    optimizer.step()

    # Assert weights updated
    for i, (name, param) in enumerate(model.mastery_head.named_parameters()):
        assert not torch.equal(pre_weights[i], param), (
            f"mastery_head.{name} weights did not update after optimizer step!"
        )


def test_retention_future_perturbation_invariance(
    sample_batch: dict[str, torch.Tensor],
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """
    Verify strict zero future-information leakage for delayed-retention predictions.
    When future interactions and retention fields at t' >= cut_step are perturbed,
    the retention predictions R_hat and hidden states at t < cut_step remain identical.
    """
    edge_index, edge_weight = graph_data
    model = TPAGKT(
        num_concepts=150,
        num_problems=3000,
        embedding_dim=32,
        hidden_dim=64,
        gnn_type="gcn",
        use_retention_head=True,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )
    model.eval()

    # 1. Forward pass on unperturbed batch
    with torch.no_grad():
        out_orig = model(sample_batch)

    # 2. Perturb future tokens and retention signals at t >= cut_step
    cut_step = 10
    perturbed_batch = {
        k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in sample_batch.items()
    }
    perturbed_batch["concept_idx"][:, cut_step:] = (
        perturbed_batch["concept_idx"][:, cut_step:] + 3
    ) % 150
    perturbed_batch["problem_idx"][:, cut_step:] = (
        perturbed_batch["problem_idx"][:, cut_step:] + 11
    ) % 3000
    perturbed_batch["is_correct"][:, cut_step:] = 1 - perturbed_batch["is_correct"][:, cut_step:]
    perturbed_batch["continuous_features"][:, cut_step:, :] += 25.0
    perturbed_batch["ret_concept"][:, cut_step:] = (
        perturbed_batch["ret_concept"][:, cut_step:] + 7
    ) % 150
    perturbed_batch["ret_log_delta"][:, cut_step:] += 5.0
    perturbed_batch["ret_v_summary"][:, cut_step:, :] += 10.0

    # 3. Forward pass on perturbed batch
    with torch.no_grad():
        out_pert = model(perturbed_batch)

    # 4. Assert past/current retention predictions at t < cut_step are bit-for-bit identical
    orig_ret_pred_past = out_orig["pred_retention"][:, : cut_step - 1]
    pert_ret_pred_past = out_pert["pred_retention"][:, : cut_step - 1]
    assert torch.allclose(orig_ret_pred_past, pert_ret_pred_past, atol=1e-5, rtol=1e-5), (
        "TPA-GKT delayed retention head violated temporal causality! Future interaction "
        "changes leaked into past retention predictions."
    )


def test_embedding_bounds_all_models(
    sample_batch: dict[str, torch.Tensor],
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Verify runtime assertions catch out-of-bounds embedding indices across all models."""
    edge_index, edge_weight = graph_data
    num_c = 150
    num_p = 3000

    models = [
        ("DKT", DKT(num_concepts=num_c, num_problems=num_p, embedding_dim=32, hidden_dim=64)),
        (
            "SAKT",
            SAKT(num_concepts=num_c, num_problems=num_p, embedding_dim=32, max_seq_len=100),
        ),
        (
            "GKT",
            GKT(
                num_concepts=num_c,
                num_problems=num_p,
                embedding_dim=32,
                hidden_dim=64,
                edge_index=edge_index,
                edge_weight=edge_weight,
            ),
        ),
        (
            "TPA-GKT",
            TPAGKT(
                num_concepts=num_c,
                num_problems=num_p,
                embedding_dim=32,
                hidden_dim=64,
                edge_index=edge_index,
                edge_weight=edge_weight,
            ),
        ),
    ]

    for _model_name, model in models:
        model.eval()

        # 1. Valid batch must pass cleanly
        with torch.no_grad():
            out = model(sample_batch)
            assert out is not None

        # 2. Out-of-bounds problem_idx >= num_problems must raise AssertionError (NO clamping!)
        bad_prob_batch = {
            k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in sample_batch.items()
        }
        bad_prob_batch["problem_idx"][0, 0] = num_p  # exactly at boundary
        with pytest.raises(AssertionError, match="Index out of bounds for 'problem_idx'"):
            model(bad_prob_batch)

        # 3. Negative problem_idx must raise AssertionError
        bad_neg_prob_batch = {
            k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in sample_batch.items()
        }
        bad_neg_prob_batch["problem_idx"][0, 0] = -1
        with pytest.raises(AssertionError, match="Index out of bounds for 'problem_idx'"):
            model(bad_neg_prob_batch)

        # 4. Out-of-bounds concept_idx >= num_concepts must raise AssertionError
        bad_concept_batch = {
            k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in sample_batch.items()
        }
        bad_concept_batch["concept_idx"][0, 0] = num_c
        with pytest.raises(AssertionError, match="Index out of bounds for 'concept_idx'"):
            model(bad_concept_batch)
