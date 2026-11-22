"""
Temporal Prerequisite-Aware Graph Knowledge Tracing (TPA-GKT)
Dataset Module: Sequence Slicing, Padding Collator, and PyTorch DataLoader

Loads preprocessed interaction parquet files, slices long student histories into
memory-safe windows (max_seq_len), and generates padded tensor batches with
aligned next-step response targets and delayed-retention evaluation targets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class KTSequenceDataset(Dataset):
    """
    PyTorch Dataset for sequential Knowledge Tracing interactions.

    Extracts student interaction sequences, slices sequences into windows of
    length <= max_seq_len, and prepares aligned next-step and retention targets.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        max_seq_len: int = 200,
        min_seq_len: int = 2,
    ) -> None:
        super().__init__()
        self.parquet_path = Path(parquet_path)
        self.max_seq_len = int(max_seq_len)
        self.min_seq_len = int(min_seq_len)

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")

        self.samples: list[dict[str, np.ndarray]] = []
        self._load_and_window_sequences()

    def _load_and_window_sequences(self) -> None:
        """Load parquet file and slice each student into windows <= max_seq_len."""
        df = pd.read_parquet(self.parquet_path)

        # Ensure deterministic chronological order
        df = df.sort_values(by=["uuid", "timestamp_sec", "orig_idx"]).reset_index(drop=True)

        # Pre-extract required columns as numpy arrays for high-speed indexing
        continuous_features = np.column_stack(
            [
                df["norm_delta_t"].values.astype(np.float32),
                df["norm_log_sec"].values.astype(np.float32),
                df["norm_log_hint_cnt"].values.astype(np.float32),
                df["norm_log_attempt"].values.astype(np.float32),
            ]
        )

        # Practice summary features for retention head v_{u,c,t}
        v_summary_features = np.column_stack(
            [
                np.log1p(df["prior_concept_attempts"].values).astype(np.float32),
                df["prior_concept_accuracy"].values.astype(np.float32),
                df["log_time_since_prev_practice"].values.astype(np.float32),
            ]
        )

        # Group by student uuid
        grouped_indices = df.groupby("uuid", sort=False).indices

        for _uuid, indices in grouped_indices.items():
            seq_len = len(indices)
            if seq_len < self.min_seq_len:
                continue

            # Slice into windows of length <= max_seq_len
            for start_idx in range(0, seq_len, self.max_seq_len):
                end_idx = min(start_idx + self.max_seq_len, seq_len)
                sub_indices = indices[start_idx:end_idx]

                if len(sub_indices) < self.min_seq_len:
                    # If remaining tail has length 1 and there is a preceding window, skip it
                    continue

                sample = {
                    "uuid": str(_uuid),
                    "concept_idx": df["concept_idx"].values[sub_indices].astype(np.int64),
                    "problem_idx": df["problem_idx"].values[sub_indices].astype(np.int64),
                    "is_correct": df["is_correct"].values[sub_indices].astype(np.int64),
                    "difficulty_id": df["difficulty_id"].values[sub_indices].astype(np.int64),
                    "is_hint_used": df["is_hint_used"].values[sub_indices].astype(np.int64),
                    "continuous_features": continuous_features[sub_indices],
                    # Aligned delayed retention fields
                    "has_retention_target": (
                        df["has_retention_target"].values[sub_indices].astype(bool)
                    ),
                    "retention_target": (
                        df["retention_target"].values[sub_indices].astype(np.float32)
                    ),
                    "retention_delta_sec": (
                        df["retention_delta_sec"].values[sub_indices].astype(np.float32)
                    ),
                    "retention_bucket_id": (
                        df["retention_bucket_id"].values[sub_indices].astype(np.int64)
                    ),
                    "v_summary": v_summary_features[sub_indices],
                }
                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return self.samples[idx]


def collate_kt_batch(batch: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    """
    Collate variable-length interaction windows into padded tensor batches.

    Returns:
      - mask: [B, L] boolean mask (True for real interactions, False for pad)
      - concept_idx: [B, L]
      - problem_idx: [B, L]
      - is_correct: [B, L]
      - difficulty_id: [B, L]
      - is_hint_used: [B, L]
      - continuous_features: [B, L, 4]
      - next_concept: [B, L] (concept query at t+1)
      - next_problem: [B, L] (problem query at t+1)
      - next_correct: [B, L] (ground-truth label at t+1)
      - next_mask: [B, L] boolean mask for valid next-step response predictions
      - ret_concept: [B, L] concept queried for retention
      - ret_log_delta: [B, L] log(1 + delta) elapsed delay
      - ret_v_summary: [B, L, 3] practice summary history
      - ret_target: [B, L] retention ground-truth label
      - ret_bucket: [B, L] retention delay bucket ID (0..3)
      - ret_mask: [B, L] boolean mask for valid retention targets
    """
    batch_size = len(batch)
    lengths = [len(item["concept_idx"]) for item in batch]
    max_len = max(lengths)

    # Pre-allocate zero/false padded batch tensors
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    concept_idx = torch.zeros((batch_size, max_len), dtype=torch.long)
    problem_idx = torch.zeros((batch_size, max_len), dtype=torch.long)
    is_correct = torch.zeros((batch_size, max_len), dtype=torch.long)
    difficulty_id = torch.zeros((batch_size, max_len), dtype=torch.long)
    is_hint_used = torch.zeros((batch_size, max_len), dtype=torch.long)
    continuous_features = torch.zeros((batch_size, max_len, 4), dtype=torch.float32)

    # Next-step prediction targets
    next_concept = torch.zeros((batch_size, max_len), dtype=torch.long)
    next_problem = torch.zeros((batch_size, max_len), dtype=torch.long)
    next_correct = torch.zeros((batch_size, max_len), dtype=torch.float32)
    next_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    # Retention prediction targets
    ret_concept = torch.zeros((batch_size, max_len), dtype=torch.long)
    ret_log_delta = torch.zeros((batch_size, max_len), dtype=torch.float32)
    ret_v_summary = torch.zeros((batch_size, max_len, 3), dtype=torch.float32)
    ret_target = torch.zeros((batch_size, max_len), dtype=torch.float32)
    ret_bucket = torch.full((batch_size, max_len), -1, dtype=torch.long)
    ret_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, item in enumerate(batch):
        seq_len = lengths[i]

        mask[i, :seq_len] = True
        concept_idx[i, :seq_len] = torch.from_numpy(item["concept_idx"])
        problem_idx[i, :seq_len] = torch.from_numpy(item["problem_idx"])
        is_correct[i, :seq_len] = torch.from_numpy(item["is_correct"])
        difficulty_id[i, :seq_len] = torch.from_numpy(item["difficulty_id"])
        is_hint_used[i, :seq_len] = torch.from_numpy(item["is_hint_used"])
        continuous_features[i, :seq_len] = torch.from_numpy(item["continuous_features"])

        # Next-step targets exist for steps t in [0, seq_len - 2]
        if seq_len > 1:
            next_concept[i, : seq_len - 1] = torch.from_numpy(item["concept_idx"][1:])
            next_problem[i, : seq_len - 1] = torch.from_numpy(item["problem_idx"][1:])
            next_correct[i, : seq_len - 1] = torch.from_numpy(
                item["is_correct"][1:].astype(np.float32)
            )
            next_mask[i, : seq_len - 1] = True

        # Retention targets aligned with step t
        ret_concept[i, :seq_len] = torch.from_numpy(item["concept_idx"])
        has_ret = item["has_retention_target"]
        ret_mask[i, :seq_len] = torch.from_numpy(has_ret)

        delta_arr = np.where(has_ret, np.log1p(np.maximum(0.0, item["retention_delta_sec"])), 0.0)
        ret_log_delta[i, :seq_len] = torch.from_numpy(delta_arr.astype(np.float32))
        ret_v_summary[i, :seq_len] = torch.from_numpy(item["v_summary"])
        ret_target[i, :seq_len] = torch.from_numpy(
            np.where(has_ret, item["retention_target"], 0.0).astype(np.float32)
        )
        ret_bucket[i, :seq_len] = torch.from_numpy(
            np.where(has_ret, item["retention_bucket_id"], -1).astype(np.int64)
        )

    batch_dict = {
        "mask": mask,
        "concept_idx": concept_idx,
        "problem_idx": problem_idx,
        "is_correct": is_correct,
        "difficulty_id": difficulty_id,
        "is_hint_used": is_hint_used,
        "continuous_features": continuous_features,
        "next_concept": next_concept,
        "next_problem": next_problem,
        "next_correct": next_correct,
        "next_mask": next_mask,
        "ret_concept": ret_concept,
        "ret_log_delta": ret_log_delta,
        "ret_v_summary": ret_v_summary,
        "ret_target": ret_target,
        "ret_bucket": ret_bucket,
        "ret_mask": ret_mask,
    }
    if len(batch) > 0 and "uuid" in batch[0]:
        batch_dict["uuid"] = [item["uuid"] for item in batch]
    return batch_dict


def create_dataloader(
    parquet_path: str | Path,
    batch_size: int = 64,
    shuffle: bool = True,
    max_seq_len: int = 200,
    num_workers: int = 0,
) -> DataLoader:
    """Create a configured DataLoader for training or evaluation."""
    dataset = KTSequenceDataset(parquet_path=parquet_path, max_seq_len=max_seq_len)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_kt_batch,
        num_workers=num_workers,
    )
