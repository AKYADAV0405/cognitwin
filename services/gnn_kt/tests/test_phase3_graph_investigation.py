"""
Automated Unit Tests for Phase 3 Stage A: Graph Contribution & Validity Audit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data


@pytest.fixture
def graph_dir():
    return Path("data/processed")


@pytest.fixture
def phase3_dir():
    return Path("services/gnn_kt/outputs/phase3")


def test_hierarchical_graph_semantics_and_symmetry(graph_dir: Path):
    """Verify that A_hier is 100% symmetric, has 0 self-loops, and represents taxonomic clustering."""
    hier_path = graph_dir / "A_hier.pt"
    if not hier_path.exists():
        pytest.skip("A_hier.pt missing, skipping test.")

    data: Data = torch.load(hier_path, weights_only=False)
    src = data.edge_index[0].numpy()
    dst = data.edge_index[1].numpy()
    weights = data.edge_weight.numpy()

    # 1. No self loops
    assert (src != dst).all(), "A_hier must not contain self-loops."

    # 2. Perfect symmetry
    edge_map = {(int(s), int(d)): float(w) for s, d, w in zip(src, dst, weights, strict=False)}
    for (s, d), w in edge_map.items():
        rev = (d, s)
        assert rev in edge_map, f"A_hier is asymmetric: edge ({s}, {d}) missing reverse ({d}, {s})."
        assert abs(w - edge_map[rev]) < 1e-5, f"Weight mismatch in symmetric pair ({s}, {d})."

    # 3. Valid weight bounds [0.25, 1.0]
    assert (weights >= 0.25).all()
    assert (weights <= 1.0).all()


def test_transition_graph_directionality_and_sparsity(graph_dir: Path):
    """Verify that A_trans is directed, sparse, and has non-reciprocal weights."""
    trans_path = graph_dir / "A_trans.pt"
    if not trans_path.exists():
        pytest.skip("A_trans.pt missing, skipping test.")

    data: Data = torch.load(trans_path, weights_only=False)
    src = data.edge_index[0].numpy()
    dst = data.edge_index[1].numpy()
    weights = data.edge_weight.numpy()

    # 1. No self loops
    assert (src != dst).all(), "A_trans must not contain self-loops."

    # 2. Asymmetric transition probabilities
    edge_map = {(int(s), int(d)): float(w) for s, d, w in zip(src, dst, weights, strict=False)}
    mismatches = 0
    for (s, d), w in edge_map.items():
        rev = (d, s)
        if rev not in edge_map or abs(w - edge_map[rev]) > 1e-4:
            mismatches += 1
    assert mismatches > 0, "A_trans should not be symmetrically weighted."

    # 3. Sparsity: density should be much lower than A_hier
    density = len(src) / (150 * 149)
    assert density < 0.05, f"A_trans density unexpectedly high: {density:.4f}"


def test_combined_graph_composition_and_self_loops(graph_dir: Path):
    """Verify that A_combined contains exactly 150 identity self-loops and merges both graphs."""
    comb_path = graph_dir / "A_combined.pt"
    if not comb_path.exists():
        pytest.skip("A_combined.pt missing, skipping test.")

    data: Data = torch.load(comb_path, weights_only=False)
    src = data.edge_index[0].numpy()
    dst = data.edge_index[1].numpy()
    weights = data.edge_weight.numpy()

    # 1. Exactly 150 self-loops with weight 1.0
    self_loops = (src == dst)
    assert self_loops.sum() == 150, f"Expected 150 self loops in A_combined, got {self_loops.sum()}"
    assert (weights[self_loops] == 1.0).all(), "Self-loops must have weight 1.0."

    # 2. Total edges should exceed A_hier
    hier_data: Data = torch.load(graph_dir / "A_hier.pt", weights_only=False)
    assert len(src) > hier_data.edge_index.size(1)


def test_graph_overlap_high_containment(phase3_dir: Path):
    """Verify that empirical transition edges are largely contained in the taxonomic hierarchy."""
    metrics_path = phase3_dir / "graph_topology_metrics.json"
    if not metrics_path.exists():
        pytest.skip("graph_topology_metrics.json missing, skipping test.")

    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    overlap = metrics["graph_overlap_analysis"]
    pct_contained = overlap["pct_of_trans_contained_in_hier"]
    assert pct_contained > 90.0, f"Expected >90% transition edge containment, got {pct_contained:.2f}%"


def test_phase3_stage_a_deliverables_exist(phase3_dir: Path):
    """Assert that all required Stage A reports, metrics, and CSVs exist on disk."""
    required_files = [
        "graph_semantics_report.md",
        "graph_topology_metrics.json",
        "graph_edge_audit.csv",
        "graph_contribution_report.md",
        "graph_gradient_diagnostics.csv",
        "graph_embedding_diagnostics.csv",
        "graph_ablation_verification.json",
    ]
    for filename in required_files:
        filepath = phase3_dir / filename
        assert filepath.exists(), f"Phase 3 deliverable missing: {filepath}"
        assert filepath.stat().st_size > 0, f"Phase 3 deliverable empty: {filepath}"
