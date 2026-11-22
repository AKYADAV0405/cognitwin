"""
Phase 1 Validation Tests: Dataset, Sequence Slicing, Padding Masking & State Continuity.

Verifies:
  1. Unique interaction representation (zero duplicate weighting, zero dropped tails).
  2. Exact expected next-step target calculation: sum(max(n_u - 1, 0)).
  3. Continuous next-response target preservation across window boundaries.
  4. Final tails of length one (rebalanced into valid contiguous slices).
  5. Padding invariance & unpadded final hidden-state extraction.
  6. SAKT key_padding_mask causal attention isolation.
  7. Continuous learner-state carryover vs. independent window reset.
  8. Interaction ID (orig_idx) traceability and alignment.
  9. Synthetic end-to-end training step (forward, 3 losses, backprop, optimizer step).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from services.gnn_kt.src.baselines import SAKT
from services.gnn_kt.src.dataset import (
    KTSequenceDataset,
    collate_kt_batch,
    partition_sequence_slices,
)
from services.gnn_kt.src.model import TPAGKT


# ==============================================================================
# Helper to create synthetic interaction DataFrames
# ==============================================================================
def create_synthetic_student_df(
    student_lengths: dict[str, int],
    num_concepts: int = 10,
    num_problems: int = 20,
) -> pd.DataFrame:
    """Create a minimal deterministic DataFrame matching KTSequenceDataset schema."""
    records = []
    global_idx = 0
    t_start = 1600000000

    for uuid, n_items in student_lengths.items():
        for step in range(n_items):
            c_idx = (step + int(uuid.replace("u", ""))) % num_concepts
            p_idx = (step * 2) % num_problems
            is_corr = int((step % 3) != 0)
            records.append(
                {
                    "uuid": uuid,
                    "ucid": f"concept_{c_idx}",
                    "upid": f"problem_{p_idx}",
                    "orig_idx": global_idx,
                    "timestamp_sec": t_start + step * 60,
                    "concept_idx": c_idx,
                    "problem_idx": p_idx,
                    "is_correct": is_corr,
                    "difficulty_id": 1,
                    "is_hint_used": 0,
                    "norm_delta_t": 0.5,
                    "norm_log_sec": 1.2,
                    "norm_log_hint_cnt": 0.0,
                    "norm_log_attempt": 0.0,
                    "prior_concept_attempts": step,
                    "prior_concept_accuracy": 0.75,
                    "log_time_since_prev_practice": 2.0,
                    "has_retention_target": False,
                    "retention_target": 0.0,
                    "retention_delta_sec": 0.0,
                    "retention_bucket_id": -1,
                }
            )
            global_idx += 1

    return pd.DataFrame(records)


# ==============================================================================
# Test 1: Partition Slices & Tail Rebalancing
# ==============================================================================
def test_partition_sequence_slices_conservation() -> None:
    """Assert every interaction index is represented exactly once with zero duplicate weighting."""
    test_lengths = [1, 2, 5, 50, 199, 200, 201, 202, 399, 400, 401, 750]

    for n in test_lengths:
        slices = partition_sequence_slices(seq_len=n, max_seq_len=200, min_seq_len=2)
        total_covered = sum(end - start for start, end in slices)
        assert total_covered == n, f"Failed length conservation for n={n}: {slices}"

        # Assert contiguous and non-overlapping
        for i in range(len(slices) - 1):
            assert slices[i][1] == slices[i + 1][0], f"Gap or overlap in slices: {slices}"

        # Assert slice bounds
        for start, end in slices:
            slice_len = end - start
            assert slice_len <= 200, f"Slice length > 200: {slice_len}"
            if n >= 2:
                assert slice_len >= 2, f"Slice length < 2: {slice_len}"


def test_final_tail_of_length_one_rebalancing() -> None:
    """Assert a tail of length 1 (e.g. n=201) is rebalanced into [199, 2] instead of discarded."""
    slices = partition_sequence_slices(seq_len=201, max_seq_len=200, min_seq_len=2)
    assert len(slices) == 2
    assert slices[0] == (0, 199)
    assert slices[1] == (199, 201)
    assert sum(end - start for start, end in slices) == 201


# ==============================================================================
# Test 2: Target Continuity Across Boundaries & Expected Target Count
# ==============================================================================
def test_dataset_target_continuity_and_expected_counts() -> None:
    """
    Assert:
      1. Expected next-step targets == sum(max(n_u - 1, 0)).
      2. Boundary next-step targets match step 0 of the subsequent window.
      3. Interaction IDs (orig_idx) are preserved and unique.
    """
    student_lengths = {
        "u1": 201,  # 2 windows: [199, 2], expected targets = 200
        "u2": 50,  # 1 window: [50], expected targets = 49
        "u3": 2,  # 1 window: [2], expected targets = 1
        "u4": 401,  # 3 windows: [200, 199, 2], expected targets = 400
    }
    df = create_synthetic_student_df(student_lengths)

    with tempfile.TemporaryDirectory() as tmp_dir:
        pq_path = Path(tmp_dir) / "test_data.parquet"
        df.to_parquet(pq_path, index=False)

        dataset = KTSequenceDataset(parquet_path=pq_path, max_seq_len=200, min_seq_len=2)

        # Expected total targets: sum(max(n_u - 1, 0))
        expected_total_targets = sum(max(n - 1, 0) for n in student_lengths.values())
        actual_total_targets = sum(sample["has_next_step"].sum() for sample in dataset)
        assert actual_total_targets == expected_total_targets, (
            f"Expected {expected_total_targets} targets, got {actual_total_targets}"
        )

        # Verify boundary target alignment for u1 (201 items -> window 0 has 199 items)
        u1_samples = [s for s in dataset if s["uuid"] == "u1"]
        assert len(u1_samples) == 2
        w0, w1 = u1_samples[0], u1_samples[1]

        # Final position of window 0 (index 198) must predict position 0 of window 1
        assert bool(w0["has_next_step"][-1]) is True
        assert w0["next_concept"][-1] == w1["concept_idx"][0]
        assert w0["next_problem"][-1] == w1["problem_idx"][0]
        assert w0["next_correct"][-1] == float(w1["is_correct"][0])
        assert w0["next_orig_idx"][-1] == w1["orig_idx"][0]

        # Final position of window 1 (last interaction of student history) has no next target
        assert bool(w1["has_next_step"][-1]) is False
        assert w1["next_orig_idx"][-1] == -1


# ==============================================================================
# Test 3: Collate Batch & Traceability
# ==============================================================================
def test_collate_batch_traceability() -> None:
    """Assert orig_idx, next_orig_idx, and window metadata are collated properly."""
    student_lengths = {"u1": 10, "u2": 25}
    df = create_synthetic_student_df(student_lengths)

    with tempfile.TemporaryDirectory() as tmp_dir:
        pq_path = Path(tmp_dir) / "test_collate.parquet"
        df.to_parquet(pq_path, index=False)

        dataset = KTSequenceDataset(parquet_path=pq_path)
        batch = collate_kt_batch([dataset[0], dataset[1]])

        assert "orig_idx" in batch
        assert "next_orig_idx" in batch
        assert "window_idx" in batch
        assert "total_windows" in batch

        # Verify mask alignment
        mask = batch["mask"]
        assert mask[0, :10].all() and not mask[0, 10:].any()
        assert mask[1, :25].all()

        # Unpadded orig_idx are valid >= 0, padded are -1
        assert (batch["orig_idx"][mask] >= 0).all()
        assert (batch["orig_idx"][~mask] == -1).all()


# ==============================================================================
# Test 4: Padding Invariance in TPAGKT & Baselines
# ==============================================================================
def test_tpa_gkt_padding_invariance() -> None:
    """Assert model predictions on real timesteps are unaffected by right padding length."""
    model = TPAGKT(num_concepts=10, num_problems=20, embedding_dim=16, hidden_dim=32)
    model.eval()

    # Short batch of length 5 (no padding)
    b_short = {
        "mask": torch.ones(1, 5, dtype=torch.bool),
        "concept_idx": torch.tensor([[1, 2, 3, 4, 5]]),
        "problem_idx": torch.tensor([[2, 4, 6, 8, 10]]),
        "is_correct": torch.tensor([[1, 0, 1, 1, 0]]),
        "difficulty_id": torch.zeros(1, 5, dtype=torch.long),
        "is_hint_used": torch.zeros(1, 5, dtype=torch.long),
        "continuous_features": torch.zeros(1, 5, 4),
        "next_concept": torch.tensor([[2, 3, 4, 5, 6]]),
        "next_problem": torch.tensor([[4, 6, 8, 10, 12]]),
        "next_correct": torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0]]),
        "next_mask": torch.tensor([[True, True, True, True, False]]),
        "ret_concept": torch.tensor([[1, 2, 3, 4, 5]]),
        "ret_log_delta": torch.zeros(1, 5),
        "ret_v_summary": torch.zeros(1, 5, 3),
        "ret_target": torch.zeros(1, 5),
        "ret_bucket": torch.zeros(1, 5, dtype=torch.long),
        "ret_mask": torch.zeros(1, 5, dtype=torch.bool),
    }

    # Same sequence padded to length 20
    pad_len = 15
    b_padded = {
        "mask": torch.cat([b_short["mask"], torch.zeros(1, pad_len, dtype=torch.bool)], dim=1),
        "concept_idx": torch.cat(
            [b_short["concept_idx"], torch.zeros(1, pad_len, dtype=torch.long)], dim=1
        ),
        "problem_idx": torch.cat(
            [b_short["problem_idx"], torch.zeros(1, pad_len, dtype=torch.long)], dim=1
        ),
        "is_correct": torch.cat(
            [b_short["is_correct"], torch.zeros(1, pad_len, dtype=torch.long)], dim=1
        ),
        "difficulty_id": torch.cat(
            [b_short["difficulty_id"], torch.zeros(1, pad_len, dtype=torch.long)], dim=1
        ),
        "is_hint_used": torch.cat(
            [b_short["is_hint_used"], torch.zeros(1, pad_len, dtype=torch.long)], dim=1
        ),
        "continuous_features": torch.cat(
            [b_short["continuous_features"], torch.zeros(1, pad_len, 4)], dim=1
        ),
        "next_concept": torch.cat(
            [b_short["next_concept"], torch.zeros(1, pad_len, dtype=torch.long)], dim=1
        ),
        "next_problem": torch.cat(
            [b_short["next_problem"], torch.zeros(1, pad_len, dtype=torch.long)], dim=1
        ),
        "next_correct": torch.cat([b_short["next_correct"], torch.zeros(1, pad_len)], dim=1),
        "next_mask": torch.cat(
            [b_short["next_mask"], torch.zeros(1, pad_len, dtype=torch.bool)], dim=1
        ),
        "ret_concept": torch.cat(
            [b_short["ret_concept"], torch.zeros(1, pad_len, dtype=torch.long)], dim=1
        ),
        "ret_log_delta": torch.cat([b_short["ret_log_delta"], torch.zeros(1, pad_len)], dim=1),
        "ret_v_summary": torch.cat([b_short["ret_v_summary"], torch.zeros(1, pad_len, 3)], dim=1),
        "ret_target": torch.cat([b_short["ret_target"], torch.zeros(1, pad_len)], dim=1),
        "ret_bucket": torch.cat(
            [b_short["ret_bucket"], torch.zeros(1, pad_len, dtype=torch.long)], dim=1
        ),
        "ret_mask": torch.cat(
            [b_short["ret_mask"], torch.zeros(1, pad_len, dtype=torch.bool)], dim=1
        ),
    }

    with torch.no_grad():
        out_short = model(b_short)
        out_padded = model(b_padded)

    # Predictions on unpadded positions (0..4) must be virtually identical (< 1e-5)
    pred_short = out_short["pred_response"][0, :4].numpy()
    pred_padded = out_padded["pred_response"][0, :4].numpy()
    np.testing.assert_allclose(pred_short, pred_padded, rtol=1e-5, atol=1e-5)

    # Hidden states at padded positions must be zeroed
    assert (out_padded["hidden_states"][0, 5:] == 0.0).all()

    # Final hidden states must match
    np.testing.assert_allclose(
        out_short["final_hidden_state"].numpy(),
        out_padded["final_hidden_state"].numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


def test_sakt_key_padding_mask() -> None:
    """Assert SAKT causal multihead attention with key_padding_mask isolates real tokens."""
    model = SAKT(num_concepts=10, num_problems=20, embedding_dim=16, max_seq_len=50)
    model.eval()

    b_short = {
        "mask": torch.ones(1, 4, dtype=torch.bool),
        "concept_idx": torch.tensor([[1, 2, 3, 4]]),
        "problem_idx": torch.tensor([[1, 2, 3, 4]]),
        "is_correct": torch.tensor([[1, 0, 1, 0]]),
        "next_concept": torch.tensor([[2, 3, 4, 5]]),
        "next_problem": torch.tensor([[2, 3, 4, 5]]),
        "next_correct": torch.tensor([[0.0, 1.0, 0.0, 1.0]]),
        "next_mask": torch.tensor([[True, True, True, False]]),
    }

    pad_len = 10
    b_padded = {
        "mask": torch.cat([b_short["mask"], torch.zeros(1, pad_len, dtype=torch.bool)], dim=1),
        "concept_idx": torch.cat(
            [b_short["concept_idx"], torch.zeros(1, pad_len, dtype=torch.long)], dim=1
        ),
        "problem_idx": torch.cat(
            [b_short["problem_idx"], torch.zeros(1, pad_len, dtype=torch.long)], dim=1
        ),
        "is_correct": torch.cat(
            [b_short["is_correct"], torch.zeros(1, pad_len, dtype=torch.long)], dim=1
        ),
        "next_concept": torch.cat(
            [b_short["next_concept"], torch.zeros(1, pad_len, dtype=torch.long)], dim=1
        ),
        "next_problem": torch.cat(
            [b_short["next_problem"], torch.zeros(1, pad_len, dtype=torch.long)], dim=1
        ),
        "next_correct": torch.cat([b_short["next_correct"], torch.zeros(1, pad_len)], dim=1),
        "next_mask": torch.cat(
            [b_short["next_mask"], torch.zeros(1, pad_len, dtype=torch.bool)], dim=1
        ),
    }

    with torch.no_grad():
        out_short = model(b_short)
        out_padded = model(b_padded)

    pred_short = out_short["pred_response"][0, :3].numpy()
    pred_padded = out_padded["pred_response"][0, :3].numpy()
    np.testing.assert_allclose(pred_short, pred_padded, rtol=1e-5, atol=1e-5)


# ==============================================================================
# Test 5: Hidden-State Carryover vs. Reset Mode
# ==============================================================================
def test_gru_continuous_state_carryover() -> None:
    """Assert hidden state carryover across sequential windows produces continuous progression."""
    model = TPAGKT(num_concepts=10, num_problems=20, embedding_dim=16, hidden_dim=32)
    model.eval()

    # Step 1: Forward window 1
    b_w1 = {
        "mask": torch.ones(1, 4, dtype=torch.bool),
        "concept_idx": torch.tensor([[1, 2, 3, 4]]),
        "problem_idx": torch.tensor([[1, 2, 3, 4]]),
        "is_correct": torch.tensor([[1, 1, 0, 1]]),
        "difficulty_id": torch.zeros(1, 4, dtype=torch.long),
        "is_hint_used": torch.zeros(1, 4, dtype=torch.long),
        "continuous_features": torch.zeros(1, 4, 4),
        "next_concept": torch.tensor([[2, 3, 4, 5]]),
        "next_problem": torch.tensor([[2, 3, 4, 5]]),
        "next_correct": torch.tensor([[1.0, 0.0, 1.0, 1.0]]),
        "next_mask": torch.ones(1, 4, dtype=torch.bool),
        "ret_concept": torch.tensor([[1, 2, 3, 4]]),
        "ret_log_delta": torch.zeros(1, 4),
        "ret_v_summary": torch.zeros(1, 4, 3),
        "ret_target": torch.zeros(1, 4),
        "ret_bucket": torch.zeros(1, 4, dtype=torch.long),
        "ret_mask": torch.zeros(1, 4, dtype=torch.bool),
    }

    b_w2 = {
        "mask": torch.ones(1, 3, dtype=torch.bool),
        "concept_idx": torch.tensor([[5, 6, 7]]),
        "problem_idx": torch.tensor([[5, 6, 7]]),
        "is_correct": torch.tensor([[1, 0, 1]]),
        "difficulty_id": torch.zeros(1, 3, dtype=torch.long),
        "is_hint_used": torch.zeros(1, 3, dtype=torch.long),
        "continuous_features": torch.zeros(1, 3, 4),
        "next_concept": torch.tensor([[6, 7, 8]]),
        "next_problem": torch.tensor([[6, 7, 8]]),
        "next_correct": torch.tensor([[0.0, 1.0, 0.0]]),
        "next_mask": torch.tensor([[True, True, False]]),
        "ret_concept": torch.tensor([[5, 6, 7]]),
        "ret_log_delta": torch.zeros(1, 3),
        "ret_v_summary": torch.zeros(1, 3, 3),
        "ret_target": torch.zeros(1, 3),
        "ret_bucket": torch.zeros(1, 3, dtype=torch.long),
        "ret_mask": torch.zeros(1, 3, dtype=torch.bool),
    }

    with torch.no_grad():
        out_w1 = model(b_w1)
        h1_final = out_w1["final_hidden_state"]

        # Run window 2 in reset mode (default)
        out_w2_reset = model(b_w2, initial_state=None)

        # Run window 2 in continuous carryover mode
        out_w2_carry = model(b_w2, initial_state=h1_final)

    assert out_w2_reset["gru_state_mode"] == "independent_window_reset"
    assert out_w2_carry["gru_state_mode"] == "continuous_carryover"

    # Hidden states and predictions in carryover mode must differ from cold-start reset
    assert not torch.allclose(out_w2_reset["hidden_states"], out_w2_carry["hidden_states"])
    assert not torch.allclose(out_w2_reset["pred_response"], out_w2_carry["pred_response"])


# ==============================================================================
# Test 6: Small Synthetic End-to-End Experiment (Forward, 3 Losses, Optimizer Step)
# ==============================================================================
def test_synthetic_end_to_end_training_step() -> None:
    """Verify forward pass, all 3 losses, backprop and optimizer update on synthetic data."""
    torch.manual_seed(42)
    model = TPAGKT(num_concepts=8, num_problems=16, embedding_dim=16, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_weight = torch.ones(4, dtype=torch.float32)

    batch = {
        "mask": torch.ones(2, 6, dtype=torch.bool),
        "orig_idx": torch.arange(12).reshape(2, 6),
        "concept_idx": torch.randint(0, 8, (2, 6)),
        "problem_idx": torch.randint(0, 16, (2, 6)),
        "is_correct": torch.randint(0, 2, (2, 6)),
        "difficulty_id": torch.zeros(2, 6, dtype=torch.long),
        "is_hint_used": torch.zeros(2, 6, dtype=torch.long),
        "continuous_features": torch.randn(2, 6, 4),
        "next_concept": torch.randint(0, 8, (2, 6)),
        "next_problem": torch.randint(0, 16, (2, 6)),
        "next_correct": torch.randint(0, 2, (2, 6)).float(),
        "next_mask": torch.ones(2, 6, dtype=torch.bool),
        "ret_concept": torch.randint(0, 8, (2, 6)),
        "ret_log_delta": torch.randn(2, 6).abs(),
        "ret_v_summary": torch.randn(2, 6, 3),
        "ret_target": torch.randint(0, 2, (2, 6)).float(),
        "ret_bucket": torch.zeros(2, 6, dtype=torch.long),
        "ret_mask": torch.ones(2, 6, dtype=torch.bool),
    }

    # Record initial weights
    w_before = model.input_proj[0].weight.clone()

    optimizer.zero_grad()
    out = model(batch, edge_index=edge_index, edge_weight=edge_weight)

    assert out["loss_response"] > 0
    assert out["loss_mastery"] > 0
    assert out["loss_retention"] > 0
    assert out["loss"] > 0

    out["loss"].backward()
    optimizer.step()

    w_after = model.input_proj[0].weight
    diff = (w_after - w_before).abs().sum().item()
    assert diff > 1e-4, f"Parameters did not update after backprop: diff={diff}"
