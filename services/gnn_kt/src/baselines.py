"""
Temporal Prerequisite-Aware Graph Knowledge Tracing (TPA-GKT)
Baselines Module: Reproducible Fair Baseline Implementations (DKT, SAKT, GKT)

All baselines consume the exact same DataLoader batch format for strict, fair comparison.
Models:
  1. DKT  : Deep Knowledge Tracing (Piech et al.) via GRU sequence modeling.
  2. SAKT : Self-Attentive Knowledge Tracing (Pandey & Karypis) with strict causal attention mask.
  3. GKT  : Simplified Graph Knowledge Tracing (Nakagawa et al.) without delayed-retention head.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# ==============================================================================
# Baseline 1: DKT (Deep Knowledge Tracing)
# ==============================================================================
class DKT(nn.Module):
    """Deep Knowledge Tracing (DKT) with GRU backbone."""

    def __init__(
        self,
        num_concepts: int = 150,
        num_problems: int = 3000,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_rnn_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_concepts = int(num_concepts)
        self.num_problems = int(num_problems)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)

        self.concept_emb = nn.Embedding(self.num_concepts, self.embedding_dim)
        self.problem_emb = nn.Embedding(self.num_problems, self.embedding_dim, padding_idx=0)
        self.correct_emb = nn.Embedding(2, 16)

        # Input dimension: concept(d) + problem(d) + correctness(16)
        in_dim = self.embedding_dim * 2 + 16
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
            dropout=dropout if num_rnn_layers > 1 else 0.0,
        )

        # Next-response prediction head [h_t; c_{t+1}; q_{t+1}]
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.embedding_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device

        concept_idx = batch["concept_idx"].to(device)
        problem_idx = batch["problem_idx"].to(device)
        is_correct = batch["is_correct"].to(device)
        next_concept = batch["next_concept"].to(device)
        next_problem = batch["next_problem"].to(device)
        next_correct = batch["next_correct"].to(device)
        next_mask = batch["next_mask"].to(device)

        e_c = self.concept_emb(concept_idx)
        e_q = self.problem_emb(problem_idx)
        e_r = self.correct_emb(is_correct)

        x = self.input_proj(torch.cat([e_c, e_q, e_r], dim=-1))
        h, _ = self.gru(x)

        # Next-step query features
        next_c_emb = self.concept_emb(next_concept)
        next_q_emb = self.problem_emb(next_problem)

        logits = self.head(torch.cat([h, next_c_emb, next_q_emb], dim=-1)).squeeze(-1)

        loss = torch.tensor(0.0, device=device)
        if next_mask.any():
            loss = F.binary_cross_entropy_with_logits(logits[next_mask], next_correct[next_mask])

        return {
            "loss": loss,
            "loss_response": loss,
            "logits_response": logits,
            "pred_response": torch.sigmoid(logits),
            "hidden_states": h,
        }


# ==============================================================================
# Baseline 2: SAKT (Self-Attentive Knowledge Tracing)
# ==============================================================================
class SAKT(nn.Module):
    """
    Self-Attentive Knowledge Tracing (SAKT).

    Enforces strict upper-triangular causal attention masking so timestep t
    never attends to any future token t' > t (zero leakage guarantee).
    """

    def __init__(
        self,
        num_concepts: int = 150,
        num_problems: int = 3000,
        embedding_dim: int = 64,
        num_heads: int = 4,
        max_seq_len: int = 200,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_concepts = int(num_concepts)
        self.num_problems = int(num_problems)
        self.embedding_dim = int(embedding_dim)
        self.max_seq_len = int(max_seq_len)

        self.concept_emb = nn.Embedding(self.num_concepts, self.embedding_dim)
        self.problem_emb = nn.Embedding(self.num_problems, self.embedding_dim, padding_idx=0)
        self.correct_emb = nn.Embedding(2, 16)

        # Interaction projection to model dimension
        self.interact_proj = nn.Linear(self.embedding_dim * 2 + 16, self.embedding_dim)
        self.query_proj = nn.Linear(self.embedding_dim * 2, self.embedding_dim)

        self.pos_emb = nn.Embedding(self.max_seq_len, self.embedding_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(self.embedding_dim)

        self.ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
        )
        self.norm2 = nn.LayerNorm(self.embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Linear(self.embedding_dim, 1)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device

        concept_idx = batch["concept_idx"].to(device)
        problem_idx = batch["problem_idx"].to(device)
        is_correct = batch["is_correct"].to(device)
        next_concept = batch["next_concept"].to(device)
        next_problem = batch["next_problem"].to(device)
        next_correct = batch["next_correct"].to(device)
        next_mask = batch["next_mask"].to(device)

        b, seq_len = concept_idx.shape
        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(b, seq_len)
        pos_embedding = self.pos_emb(pos)

        # Interaction representation (Keys and Values)
        e_c = self.concept_emb(concept_idx)
        e_q = self.problem_emb(problem_idx)
        e_r = self.correct_emb(is_correct)
        interact = self.interact_proj(torch.cat([e_c, e_q, e_r], dim=-1)) + pos_embedding

        # Query representation (Next interaction prompt)
        next_c_emb = self.concept_emb(next_concept)
        next_q_emb = self.problem_emb(next_problem)
        query = self.query_proj(torch.cat([next_c_emb, next_q_emb], dim=-1)) + pos_embedding

        # Strict Causal Attention Mask: upper triangular elements set to -inf
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
        )

        attn_out, _ = self.attn(
            query=query,
            key=interact,
            value=interact,
            attn_mask=causal_mask,
        )
        x = self.norm1(query + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        logits = self.head(x).squeeze(-1)

        loss = torch.tensor(0.0, device=device)
        if next_mask.any():
            loss = F.binary_cross_entropy_with_logits(logits[next_mask], next_correct[next_mask])

        return {
            "loss": loss,
            "loss_response": loss,
            "logits_response": logits,
            "pred_response": torch.sigmoid(logits),
            "hidden_states": x,
        }


# ==============================================================================
# Baseline 3: GKT (Graph Knowledge Tracing)
# ==============================================================================
class GKT(nn.Module):
    """Simplified Graph Knowledge Tracing (GKT) baseline without retention head."""

    def __init__(
        self,
        num_concepts: int = 150,
        num_problems: int = 3000,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        edge_index: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_concepts = int(num_concepts)
        self.num_problems = int(num_problems)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)

        self.concept_base_emb = nn.Parameter(torch.empty(self.num_concepts, self.embedding_dim))
        nn.init.xavier_uniform_(self.concept_base_emb)

        self.gnn1 = GCNConv(self.embedding_dim, self.embedding_dim)
        self.gnn2 = GCNConv(self.embedding_dim, self.embedding_dim)
        self.gnn_norm = nn.LayerNorm(self.embedding_dim)
        self.dropout = nn.Dropout(dropout)

        if edge_index is not None:
            self.register_buffer("edge_index", edge_index)
        else:
            self.register_buffer("edge_index", torch.empty((2, 0), dtype=torch.long))

        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight)
        else:
            self.register_buffer("edge_weight", torch.empty(0, dtype=torch.float32))

        self.problem_emb = nn.Embedding(self.num_problems, self.embedding_dim, padding_idx=0)
        self.correct_emb = nn.Embedding(2, 16)

        in_dim = self.embedding_dim * 2 + 16
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.embedding_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def load_graph_from_disk(self, processed_dir: str | Path) -> None:
        target_file = Path(processed_dir) / "A_combined.pt"
        if target_file.exists():
            data: Data = torch.load(target_file, weights_only=False)
            self.edge_index = data.edge_index
            if hasattr(data, "edge_weight") and data.edge_weight is not None:
                self.edge_weight = data.edge_weight
            else:
                self.edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.float32)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        edge_index: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device

        e_idx = edge_index if edge_index is not None else self.edge_index
        e_w = edge_weight if edge_weight is not None else self.edge_weight

        # Graph encoding
        h0 = self.concept_base_emb
        if e_idx.numel() > 0:
            if e_w.numel() > 0:
                h1 = self.gnn1(h0, e_idx, e_w)
                h2 = self.gnn2(h1, e_idx, e_w)
            else:
                h1 = self.gnn1(h0, e_idx)
                h2 = self.gnn2(h1, e_idx)
            g = self.gnn_norm(h0 + self.dropout(F.relu(h2)))
        else:
            g = h0

        concept_idx = batch["concept_idx"].to(device)
        problem_idx = batch["problem_idx"].to(device)
        is_correct = batch["is_correct"].to(device)
        next_concept = batch["next_concept"].to(device)
        next_problem = batch["next_problem"].to(device)
        next_correct = batch["next_correct"].to(device)
        next_mask = batch["next_mask"].to(device)

        e_c = g[concept_idx]
        e_q = self.problem_emb(problem_idx)
        e_r = self.correct_emb(is_correct)

        x = self.input_proj(torch.cat([e_c, e_q, e_r], dim=-1))
        h, _ = self.gru(x)

        next_c_emb = g[next_concept]
        next_q_emb = self.problem_emb(next_problem)

        logits = self.head(torch.cat([h, next_c_emb, next_q_emb], dim=-1)).squeeze(-1)

        loss = torch.tensor(0.0, device=device)
        if next_mask.any():
            loss = F.binary_cross_entropy_with_logits(logits[next_mask], next_correct[next_mask])

        return {
            "loss": loss,
            "loss_response": loss,
            "logits_response": logits,
            "pred_response": torch.sigmoid(logits),
            "hidden_states": h,
            "concept_graph_emb": g,
        }
