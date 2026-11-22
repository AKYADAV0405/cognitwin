"""
Temporal Prerequisite-Aware Graph Knowledge Tracing (TPA-GKT)
Graph Builder Module: Dual Knowledge Graph Construction

Builds:
  1. A_hier.pt    : Curriculum hierarchy graph from Info_Content.csv (level4/level3 shared subtrees)
  2. A_trans.pt   : Empirical transition graph strictly from train.parquet (Zero leakage)
  3. A_combined.pt: Linear combination A_comb = alpha * A_hier + (1 - alpha) * A_trans + I

Paper Scope: Section IV.D (Dual Graph Formulation)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Data


def resolve_path(path_str: str, base_dirs: list[Path]) -> Path:
    """Resolve a path against candidate base directories."""
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    for base in base_dirs:
        candidate = (base / p).resolve()
        if candidate.exists():
            return candidate
    return (base_dirs[0] / p).resolve()


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and parse YAML configuration file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_hierarchical_graph(
    info_content_csv: Path,
    concept2idx: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construct prerequisite curriculum hierarchy graph A_hier from Info_Content.csv.

    Concepts sharing the same level4_id (or level3_id if level4_id is absent)
    are connected with bidirectional edges of weight 1.0.
    """
    content_df = pd.read_csv(info_content_csv, usecols=["ucid", "level3_id", "level4_id"])

    # Filter to concepts in current vocabulary
    tracked_concepts = set(concept2idx.keys())
    content_df = content_df[content_df["ucid"].isin(tracked_concepts)].copy()

    # Determine grouping taxonomy level per concept
    has_l4 = content_df["level4_id"].notna() & (content_df["level4_id"] != "")
    content_df["cluster_id"] = np.where(has_l4, content_df["level4_id"], content_df["level3_id"])

    # Group concept indices by taxonomy cluster
    cluster_groups = defaultdict(list)
    for _, row in content_df.iterrows():
        c_ucid = str(row["ucid"])
        if c_ucid in concept2idx:
            idx = concept2idx[c_ucid]
            cluster = str(row["cluster_id"])
            if cluster and cluster.lower() != "nan":
                cluster_groups[cluster].append(idx)

    # Generate bidirectional edges between concepts sharing cluster
    edge_set: set[tuple[int, int]] = set()
    for _cluster, members in cluster_groups.items():
        if len(members) > 1:
            for i in range(len(members)):
                for j in range(len(members)):
                    if i != j:
                        edge_set.add((members[i], members[j]))

    if edge_set:
        edges = sorted(list(edge_set))
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_weight = torch.ones(len(edges), dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty(0, dtype=torch.float32)

    return edge_index, edge_weight


def build_transition_graph(
    train_parquet: Path,
    num_nodes: int,
    transition_threshold: float = 0.05,
    min_transition_count: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, dict[tuple[int, int], float]]:
    """
    Construct empirical concept transition graph A_trans strictly from train.parquet.

    Counts consecutive distinct concept transitions T_ij, computes P(c_j | c_i) = T_ij / sum_k T_ik,
    and retains edges where P(c_j | c_i) > transition_threshold and T_ij >= min_transition_count.
    """
    cols = ["uuid", "concept_idx", "timestamp_sec", "orig_idx"]
    train_df = pd.read_parquet(train_parquet, columns=cols)

    # Ensure chronological order
    train_df = train_df.sort_values(by=["uuid", "timestamp_sec", "orig_idx"]).reset_index(
        drop=True
    )

    # Count consecutive distinct concept transitions: T_ij
    transition_counts: dict[tuple[int, int], int] = defaultdict(int)
    row_sums: dict[int, int] = defaultdict(int)

    for _, group in train_df.groupby("uuid"):
        concept_sequence = group["concept_idx"].values
        for t in range(len(concept_sequence) - 1):
            src_c = int(concept_sequence[t])
            dst_c = int(concept_sequence[t + 1])
            # Only record distinct transitions between different knowledge components
            if src_c != dst_c and 0 <= src_c < num_nodes and 0 <= dst_c < num_nodes:
                transition_counts[(src_c, dst_c)] += 1
                row_sums[src_c] += 1

    # Filter directed edges based on threshold and min count
    valid_transitions: dict[tuple[int, int], float] = {}
    for (src, dst), count in transition_counts.items():
        total_out = row_sums[src]
        if total_out > 0:
            prob = float(count / total_out)
            if prob > transition_threshold and count >= min_transition_count:
                valid_transitions[(src, dst)] = prob

    if valid_transitions:
        sorted_edges = sorted(valid_transitions.keys())
        src = [e[0] for e in sorted_edges]
        dst = [e[1] for e in sorted_edges]
        weights = [valid_transitions[e] for e in sorted_edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty(0, dtype=torch.float32)

    return edge_index, edge_weight, valid_transitions


def build_combined_graph(
    hier_edge_index: torch.Tensor,
    hier_edge_weight: torch.Tensor,
    trans_transitions: dict[tuple[int, int], float],
    num_nodes: int,
    alpha_hier: float = 0.50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct linear combination A_comb = alpha * A_hier + (1 - alpha) * A_trans + I."""
    combined_weights: dict[tuple[int, int], float] = defaultdict(float)

    # 1. Incorporate hierarchical edges
    if hier_edge_index.numel() > 0:
        hier_src = hier_edge_index[0].tolist()
        hier_dst = hier_edge_index[1].tolist()
        hier_w = hier_edge_weight.tolist()
        for s, d, w in zip(hier_src, hier_dst, hier_w, strict=False):
            combined_weights[(int(s), int(d))] += alpha_hier * float(w)

    # 2. Incorporate transition edges
    for (s, d), prob in trans_transitions.items():
        combined_weights[(s, d)] += (1.0 - alpha_hier) * float(prob)

    # 3. Add self-loops for GCN / GAT numerical stability (weight 1.0)
    for i in range(num_nodes):
        combined_weights[(i, i)] = 1.0

    sorted_comb_edges = sorted(combined_weights.keys())
    src = [e[0] for e in sorted_comb_edges]
    dst = [e[1] for e in sorted_comb_edges]
    weights = [combined_weights[e] for e in sorted_comb_edges]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)

    return edge_index, edge_weight


def run_graph_construction(
    config_path: str = "services/gnn_kt/config.yaml",
) -> dict[str, Any]:
    """Build all three knowledge graphs and output structural metrics."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent.parent

    config_candidates = [repo_root, script_dir.parent, Path.cwd()]
    cfg_file = resolve_path(config_path, config_candidates)
    cfg = load_config(cfg_file)

    data_raw_dir = resolve_path(cfg["data_raw_dir"], [repo_root, script_dir.parent, Path.cwd()])
    data_proc_dir = resolve_path(
        cfg["data_processed_dir"], [repo_root, script_dir.parent, Path.cwd()]
    )
    outputs_dir = resolve_path(
        cfg.get("outputs_dir", "services/gnn_kt/outputs"),
        [repo_root, script_dir.parent, Path.cwd()],
    )
    outputs_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = data_proc_dir / "metadata.json"
    train_parquet = data_proc_dir / "train.parquet"
    info_content_csv = data_raw_dir / "Info_Content.csv"

    if not metadata_path.exists() or not train_parquet.exists():
        raise FileNotFoundError(
            f"Preprocessed artifacts not found in {data_proc_dir}. Run preprocess.py first!"
        )

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    concept2idx: dict[str, int] = metadata["concept2idx"]
    num_nodes = len(concept2idx)
    alpha_hier = float(cfg.get("alpha_hier", 0.50))
    trans_threshold = float(cfg.get("transition_threshold", 0.05))
    min_trans_count = int(cfg.get("min_transition_count", 10))

    print("=" * 78)
    print("TPA-GKT DUAL KNOWLEDGE GRAPH CONSTRUCTION")
    print(f"Num Concept Nodes (N) : {num_nodes}")
    print(f"Alpha Hierarchical     : {alpha_hier:.2f}")
    print(f"Transition Threshold   : {trans_threshold:.3f}")
    print(f"Min Transition Count   : {min_trans_count}")
    print("=" * 78)

    # 1. Build Curriculum Hierarchy Graph
    print("\n[1/3] Building Curriculum Hierarchy Graph (A_hier.pt)...")
    hier_edge_index, hier_edge_weight = build_hierarchical_graph(info_content_csv, concept2idx)
    data_hier = Data(
        edge_index=hier_edge_index, edge_weight=hier_edge_weight, num_nodes=num_nodes
    )
    torch.save(data_hier, data_proc_dir / "A_hier.pt")
    num_hier_edges = hier_edge_index.size(1) if hier_edge_index.numel() > 0 else 0
    print(f"  A_hier: {num_hier_edges:,} bidirectional edges")

    # 2. Build Empirical Transition Graph (Zero Val/Test Leakage!)
    print("\n[2/3] Building Inferred Transition Graph (A_trans.pt) STRICTLY on Train split...")
    trans_edge_index, trans_edge_weight, trans_dict = build_transition_graph(
        train_parquet,
        num_nodes=num_nodes,
        transition_threshold=trans_threshold,
        min_transition_count=min_trans_count,
    )
    data_trans = Data(
        edge_index=trans_edge_index, edge_weight=trans_edge_weight, num_nodes=num_nodes
    )
    torch.save(data_trans, data_proc_dir / "A_trans.pt")
    num_trans_edges = trans_edge_index.size(1) if trans_edge_index.numel() > 0 else 0
    print(f"  A_trans: {num_trans_edges:,} directed edges")

    # 3. Build Combined Adjacency Graph with Self-Loops
    print("\n[3/3] Building Combined Topology Graph with Self-Loops (A_combined.pt)...")
    comb_edge_index, comb_edge_weight = build_combined_graph(
        hier_edge_index,
        hier_edge_weight,
        trans_dict,
        num_nodes=num_nodes,
        alpha_hier=alpha_hier,
    )
    data_comb = Data(
        edge_index=comb_edge_index, edge_weight=comb_edge_weight, num_nodes=num_nodes
    )
    torch.save(data_comb, data_proc_dir / "A_combined.pt")
    num_comb_edges = comb_edge_index.size(1)
    density = float(num_comb_edges / (num_nodes * num_nodes))

    # Calculate isolated nodes (nodes with 0 non-self-loop edges in combined graph)
    comb_src = comb_edge_index[0].numpy()
    comb_dst = comb_edge_index[1].numpy()
    non_self_mask = comb_src != comb_dst
    active_nodes = set(comb_src[non_self_mask]).union(set(comb_dst[non_self_mask]))
    isolated_nodes = num_nodes - len(active_nodes)

    stats = {
        "num_nodes": num_nodes,
        "num_hier_edges": num_hier_edges,
        "num_trans_edges": num_trans_edges,
        "num_combined_edges": num_comb_edges,
        "density_combined": density,
        "isolated_nodes_count": isolated_nodes,
        "isolated_nodes_pct": float(isolated_nodes / num_nodes),
        "alpha_hier": alpha_hier,
        "transition_threshold": trans_threshold,
        "min_transition_count": min_trans_count,
    }

    graph_stats_file = outputs_dir / "graph_stats.json"
    with open(graph_stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 78)
    print("GRAPH TOPOLOGY SUMMARY")
    print("=" * 78)
    print(f"Nodes                   : {num_nodes}")
    print(f"Hierarchical Edges      : {num_hier_edges:,}")
    print(f"Transition Edges        : {num_trans_edges:,}")
    print(f"Combined Edges (w/ I)   : {num_comb_edges:,}")
    print(f"Combined Graph Density  : {density:.4f}")
    print(f"Isolated Nodes (non-I)  : {isolated_nodes} ({isolated_nodes / num_nodes * 100:.1f}%)")
    print(f"Saved stats to          : {graph_stats_file}")
    print("=" * 78)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TPA-GKT Graph Builder")
    parser.add_argument(
        "--config", type=str, default="services/gnn_kt/config.yaml", help="Path to config.yaml"
    )
    args = parser.parse_args()

    run_graph_construction(config_path=args.config)
