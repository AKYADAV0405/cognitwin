"""
Temporal Prerequisite-Aware Graph Knowledge Tracing (TPA-GKT)
Model Architecture Module: 4-Stage GNN-KT Model with Three Prediction Heads

Paper Scope: Section IV (Methodology & Model Architecture)
Stages:
  1. Graph Neural Network Encoder (GCN / GAT over Dual Knowledge Graph)
  2. Interaction Encoder & Temporal Recurrent State Updater (GRU)
  3. Three Prediction Heads (Next-Response, Concept Mastery, Retention Decay)
  4. Joint Multi-Task Masked Loss Formulation
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv


def _assert_embedding_bounds(name: str, indices: torch.Tensor, upper: int) -> None:
    """Validate that tensor indices are strictly non-negative and within embedding size."""
    if indices.numel() > 0:
        idx_min = int(indices.min().item())
        idx_max = int(indices.max().item())
        assert 0 <= idx_min and idx_max < upper, (
            f"Index out of bounds for '{name}': min={idx_min}, max={idx_max}, "
            f"expected in [0, {upper})"
        )


class TPAGKT(nn.Module):
    """
    Temporal Prerequisite-Aware Graph Knowledge Tracing (TPA-GKT) Model.

    Combines dual curriculum/transition topology GNN propagation, multi-modal
    interaction dynamics, and dual prediction heads (next-step correctness
    and time-conditioned delayed retention).
    """

    def __init__(
        self,
        num_concepts: int = 150,
        num_problems: int = 3000,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_gnn_layers: int = 2,
        num_rnn_layers: int = 1,
        dropout: float = 0.2,
        gat_heads: int = 4,
        loss_weight_mastery: float = 0.2,
        loss_weight_retention: float = 0.1,
        # Ablation Toggles
        use_graph: bool = True,
        graph_type: str = "combined",  # "combined" | "hier_only" | "trans_only"
        gnn_type: str = "gcn",  # "gcn" | "gat"
        use_time_decay: bool = True,
        use_attempt_hint: bool = True,
        use_retention_head: bool = True,
        # Graph data paths / tensors
        edge_index: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_concepts = int(num_concepts)
        self.num_problems = int(num_problems)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout_rate = float(dropout)
        self.loss_weight_mastery = float(loss_weight_mastery)
        self.loss_weight_retention = float(loss_weight_retention)

        # Ablation flags
        self.use_graph = bool(use_graph)
        self.graph_type = str(graph_type)
        self.gnn_type = str(gnn_type).lower()
        self.use_time_decay = bool(use_time_decay)
        self.use_attempt_hint = bool(use_attempt_hint)
        self.use_retention_head = bool(use_retention_head)

        # ----------------------------------------------------------------------
        # Stage 1: Graph Neural Network Concept Encoder
        # ----------------------------------------------------------------------
        self.concept_base_emb = nn.Parameter(torch.empty(self.num_concepts, self.embedding_dim))
        nn.init.xavier_uniform_(self.concept_base_emb)

        if self.use_graph:
            if self.gnn_type == "gat":
                self.gnn1 = GATConv(
                    self.embedding_dim,
                    self.embedding_dim // gat_heads,
                    heads=gat_heads,
                    concat=True,
                )
                self.gnn2 = GATConv(
                    self.embedding_dim,
                    self.embedding_dim,
                    heads=1,
                    concat=False,
                )
            else:  # Default: GCN
                self.gnn1 = GCNConv(self.embedding_dim, self.embedding_dim)
                self.gnn2 = GCNConv(self.embedding_dim, self.embedding_dim)

            self.gnn_norm1 = nn.LayerNorm(self.embedding_dim)
            self.gnn_norm2 = nn.LayerNorm(self.embedding_dim)
        else:
            self.gnn_norm0 = nn.LayerNorm(self.embedding_dim)

        self.gnn_dropout = nn.Dropout(self.dropout_rate)

        # Register graph buffers
        if edge_index is not None:
            self.register_buffer("edge_index", edge_index)
        else:
            self.register_buffer("edge_index", torch.empty((2, 0), dtype=torch.long))

        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight)
        else:
            self.register_buffer("edge_weight", torch.empty(0, dtype=torch.float32))

        # ----------------------------------------------------------------------
        # Stage 2: Multi-Modal Interaction Encoder & Recurrent Updater
        # ----------------------------------------------------------------------
        self.problem_emb = nn.Embedding(self.num_problems, self.embedding_dim, padding_idx=0)
        self.correct_emb = nn.Embedding(2, 16)
        self.diff_emb = nn.Embedding(4, 16)
        self.hint_emb = nn.Embedding(2, 8)
        self.cont_proj = nn.Linear(4, 16)

        # Input dimension: problem(d) + concept_graph(d) + correct(16) +
        # diff(16) + hint(8) + cont(16)
        input_feature_dim = self.embedding_dim * 2 + 16 + 16 + 8 + 16
        self.input_proj = nn.Sequential(
            nn.Linear(input_feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )

        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
            dropout=self.dropout_rate if num_rnn_layers > 1 else 0.0,
        )

        # ----------------------------------------------------------------------
        # Stage 3: Three Prediction Heads
        # ----------------------------------------------------------------------
        # Head 1: Next-Response Correctness Head [h_{u,t}; g_{c_{t+1}}; e_{q_{t+1}}]
        self.response_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.embedding_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 1),
        )

        # Head 2: Concept Mastery Head [h_{u,t}; g_c]
        self.mastery_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.embedding_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        # Head 3: Time-Conditioned Delayed-Retention Head [h_{u,t}; g_c; log(1+delta); v_{u,c,t}]
        self.retention_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.embedding_dim + 1 + 3, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 1),
        )

    def load_graph_from_disk(self, processed_dir: str | Path) -> None:
        """Load graph topology tensor matching graph_type from data/processed/."""
        p_dir = Path(processed_dir)
        filename_map = {
            "combined": "A_combined.pt",
            "hier_only": "A_hier.pt",
            "trans_only": "A_trans.pt",
        }
        target_file = p_dir / filename_map.get(self.graph_type, "A_combined.pt")
        if target_file.exists():
            data: Data = torch.load(target_file, weights_only=False)
            self.edge_index = data.edge_index
            if hasattr(data, "edge_weight") and data.edge_weight is not None:
                self.edge_weight = data.edge_weight
            else:
                self.edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.float32)

    def encode_graph(
        self,
        edge_index: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Stage 1: Encode concept graph topology via GCN / GAT.

        Returns:
          g: [num_concepts, embedding_dim] graph-propagated concept representations.
        """
        h0 = self.concept_base_emb

        if not self.use_graph:
            return self.gnn_norm0(h0)

        e_idx = edge_index if edge_index is not None else self.edge_index
        e_w = edge_weight if edge_weight is not None else self.edge_weight

        if e_idx.numel() == 0:
            return h0

        if self.gnn_type == "gat":
            h1 = self.gnn1(h0, e_idx)
            h1 = self.gnn_norm1(h0 + self.gnn_dropout(F.elu(h1)))
            h2 = self.gnn2(h1, e_idx)
            g = self.gnn_norm2(h1 + self.gnn_dropout(F.elu(h2)))
        else:  # GCN
            if e_w.numel() > 0:
                h1 = self.gnn1(h0, e_idx, e_w)
            else:
                h1 = self.gnn1(h0, e_idx)
            h1 = self.gnn_norm1(h0 + self.gnn_dropout(F.relu(h1)))

            if e_w.numel() > 0:
                h2 = self.gnn2(h1, e_idx, e_w)
            else:
                h2 = self.gnn2(h1, e_idx)
            g = self.gnn_norm2(h1 + self.gnn_dropout(F.relu(h2)))

        return g

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        edge_index: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Execute forward propagation across Stages 1-4.

        Returns:
          dict containing losses, predictions, and latent representations.
        """
        device = next(self.parameters()).device

        # Unpack input batch tensors
        mask = batch["mask"].to(device)
        concept_idx = batch["concept_idx"].to(device)
        problem_idx = batch["problem_idx"].to(device)
        is_correct = batch["is_correct"].to(device)
        difficulty_id = batch["difficulty_id"].to(device)
        is_hint_used = batch["is_hint_used"].to(device)
        cont_features = batch["continuous_features"].to(device)  # [B, L, 4]

        # Apply ablation toggles to interaction features
        if not self.use_time_decay:
            # Zero out norm_delta_t (feature index 0)
            cont_features = cont_features.clone()
            cont_features[:, :, 0] = 0.0

        if not self.use_attempt_hint:
            # Zero out hint indicator, hint count (idx 2), attempt count (idx 3)
            is_hint_used = torch.zeros_like(is_hint_used)
            cont_features = cont_features.clone()
            cont_features[:, :, 2] = 0.0
            cont_features[:, :, 3] = 0.0

        # Stage 1: Encode concept graph
        g = self.encode_graph(edge_index=edge_index, edge_weight=edge_weight)

        # Stage 2: Interaction Representation & Temporal Recurrent Update
        _assert_embedding_bounds("problem_idx", problem_idx, self.problem_emb.num_embeddings)
        _assert_embedding_bounds("concept_idx", concept_idx, self.num_concepts)
        _assert_embedding_bounds("is_correct", is_correct, self.correct_emb.num_embeddings)
        _assert_embedding_bounds("difficulty_id", difficulty_id, self.diff_emb.num_embeddings)
        _assert_embedding_bounds("is_hint_used", is_hint_used, self.hint_emb.num_embeddings)

        e_prob = self.problem_emb(problem_idx)
        e_concept_graph = g[concept_idx]
        e_corr = self.correct_emb(is_correct)
        e_diff = self.diff_emb(difficulty_id)
        e_hint = self.hint_emb(is_hint_used)
        e_cont = self.cont_proj(cont_features)

        interaction_repr = torch.cat(
            [e_prob, e_concept_graph, e_corr, e_diff, e_hint, e_cont], dim=-1
        )
        x = self.input_proj(interaction_repr)

        # Recurrent state progression
        h, _ = self.gru(x)  # [B, L, hidden_dim]

        # Stage 3: Head 1 — Next-Response Correctness Head
        next_concept = batch["next_concept"].to(device)
        next_problem = batch["next_problem"].to(device)
        next_correct = batch["next_correct"].to(device)
        next_mask = batch["next_mask"].to(device)

        _assert_embedding_bounds("next_concept", next_concept, self.num_concepts)
        _assert_embedding_bounds("next_problem", next_problem, self.problem_emb.num_embeddings)

        next_g = g[next_concept]
        next_p_emb = self.problem_emb(next_problem)
        response_input = torch.cat([h, next_g, next_p_emb], dim=-1)
        logits_response = self.response_head(response_input).squeeze(-1)  # [B, L]

        # Stage 3: Head 2 — Auxiliary Concept Mastery Supervision on next_concept
        mastery_input = torch.cat([h, next_g], dim=-1)
        logits_mastery = self.mastery_head(mastery_input).squeeze(-1)  # [B, L]

        # Stage 3: Head 3 — Time-Conditioned Delayed Retention Head
        ret_concept = batch["ret_concept"].to(device)
        ret_log_delta = batch["ret_log_delta"].to(device).unsqueeze(-1)  # [B, L, 1]
        ret_v_summary = batch["ret_v_summary"].to(device)  # [B, L, 3]
        ret_target = batch["ret_target"].to(device)
        ret_mask = batch["ret_mask"].to(device)

        _assert_embedding_bounds("ret_concept", ret_concept, self.num_concepts)

        ret_g = g[ret_concept]
        retention_input = torch.cat([h, ret_g, ret_log_delta, ret_v_summary], dim=-1)
        logits_retention = self.retention_head(retention_input).squeeze(-1)  # [B, L]

        # Stage 4: Masked Loss Calculation
        loss_response = torch.tensor(0.0, device=device)
        if next_mask.any():
            loss_response = F.binary_cross_entropy_with_logits(
                logits_response[next_mask], next_correct[next_mask]
            )

        loss_mastery = torch.tensor(0.0, device=device)
        if next_mask.any():
            loss_mastery = F.binary_cross_entropy_with_logits(
                logits_mastery[next_mask], next_correct[next_mask]
            )

        loss_retention = torch.tensor(0.0, device=device)
        if self.use_retention_head and ret_mask.any():
            loss_retention = F.binary_cross_entropy_with_logits(
                logits_retention[ret_mask], ret_target[ret_mask]
            )

        total_loss = loss_response + self.loss_weight_mastery * loss_mastery
        if self.use_retention_head:
            total_loss = total_loss + self.loss_weight_retention * loss_retention

        return {
            "loss": total_loss,
            "loss_response": loss_response,
            "loss_mastery": loss_mastery,
            "loss_retention": loss_retention,
            "logits_response": logits_response,
            "pred_response": torch.sigmoid(logits_response),
            "logits_mastery": logits_mastery,
            "pred_mastery": torch.sigmoid(logits_mastery),
            "logits_retention": logits_retention,
            "pred_retention": torch.sigmoid(logits_retention),
            "hidden_states": h,
            "concept_graph_emb": g,
            "mask": mask,
            "next_mask": next_mask,
            "ret_mask": ret_mask,
        }

    def compute_concept_mastery(
        self,
        h: torch.Tensor,
        g: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Stage 3: Head 2 — Compute mastery matrix M_hat_{u,c,t} for all concepts.

        Args:
          h: [B, L, hidden_dim] hidden state sequence from GRU
          g: [num_concepts, embedding_dim] graph concept embeddings

        Returns:
          mastery: [B, L, num_concepts] estimated concept mastery probabilities in [0, 1].
        """
        if g is None:
            g = self.encode_graph()

        b, seq_len, _ = h.shape
        n_concepts = g.shape[0]

        # Expand tensors to form all [B, L, N] combinations: [h_{u,t}; g_c]
        h_exp = h.unsqueeze(2).expand(b, seq_len, n_concepts, self.hidden_dim)
        g_exp = g.unsqueeze(0).unsqueeze(0).expand(b, seq_len, n_concepts, self.embedding_dim)
        mastery_input = torch.cat([h_exp, g_exp], dim=-1)

        logits_mastery = self.mastery_head(mastery_input).squeeze(-1)
        return torch.sigmoid(logits_mastery)
