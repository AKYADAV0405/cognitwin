# Graph Semantics and Topological Validity Report
> **Phase 3 Stage A: Investigation of Knowledge Graph Foundations**
> **Model**: Temporal Prerequisite-Aware Graph Knowledge Tracing (TPA-GKT)
> **Dataset**: Junyi Academy 2020 (Math Curriculum)
> **Status**: Completed Stage A Diagnostic Audit

---

## 1. Executive Summary & Semantic Clarification

A critical requirement of research rigor is ensuring that formal graph definitions in the manuscript accurately reflect the empirical objects constructed in code:

1. **Curriculum Hierarchy Graph (`A_hier`)**:
   - **Data Origin**: `Info_Content.csv` columns `level1_id`, `level2_id`, `level3_id`, `level4_id`.
   - **Semantic Nature**: **Taxonomic Co-Membership / Topic Clustering**. Nodes are connected if they share a fine-grained topic cluster (`level4_id` / `level3_id`) with fallback to `level2_id`.
   - **Directionality**: **100% Symmetrized / Bidirectional**. For every edge $(i, j)$ with weight $w$, the reverse edge $(j, i)$ is also added with identical weight.
   - **Prerequisite Validity**: **This is NOT a Directed Acyclic Graph (DAG) of prerequisite dependencies**. It contains 1 cyclic components. It represents curriculum *domain similarity*, not prerequisite precedence.

2. **Empirical Behavioral Transition Graph (`A_trans`)**:
   - **Data Origin**: Chronologically ordered interactions strictly within `train.parquet` (zero validation/test sequence leakage).
   - **Semantic Nature**: **Asymmetric Learner Progression Proxy**. Directed edges $(i 	o j)$ represent conditional probability $P(c_j \mid c_i) > 0.05$ with minimum occurrence count $\ge 10$.
   - **Directionality**: **Directed & Asymmetric** (reciprocity ratio = 0.9010). Only 182 reciprocal edge pairs exist out of 404 directed edges.

3. **Combined Topology Graph (`A_combined`)**:
   - **Formula**: $A_{\text{comb}} = \alpha \cdot A_{\text{hier}} + (1 - \alpha) \cdot A_{\text{trans}} + I$ with $\alpha = 0.50$.
   - **Edge Overlap Discovery**: **94.8% of all transition edges (383 / 404) are already contained within the hierarchical graph**.

---

## 2. Comparative Graph Topology Metrics

| Topological Metric | Curriculum Hierarchy (`A_hier`) | Empirical Transition (`A_trans`) | Combined Graph (`A_combined`) |
|:---|---:|---:|---:|
| **Total Directed Edges** | **3,036** | **404** | **3,207** |
| **Self-Loops** | 0 | 0 | 150 (Identity $I$) |
| **Non-Self Directed Edges** | 3,036 | 404 | 3,057 |
| **Graph Density (excl. self-loops)** | 0.1358 | 0.0181 | 0.1368 |
| **Mean In-Degree $\pm$ Std** | 20.2 $\pm$ 24.7 | 2.7 $\pm$ 1.8 | 20.4 $\pm$ 24.7 |
| **Mean Out-Degree $\pm$ Std** | 20.2 $\pm$ 24.7 | 2.7 $\pm$ 1.8 | 20.4 $\pm$ 24.7 |
| **Isolated Nodes (Degree = 0)** | 0 (0.0%) | 15 (10.0%) | 0 (0.0%) |
| **Reciprocity Ratio (Symmetry)** | **1.0000 (100% Symmetric)** | **0.9010 (Asymmetric)** | **0.9964** |
| **Is Directed Acyclic Graph (DAG)?** | **NO (False)** | **NO (False)** | **NO (False)** |
| **Weakly Connected Components** | 1 | 39 | 1 |
| **Largest Connected Component Size** | 150 / 150 | 25 / 150 | 150 / 150 |
| **Graph Diameter (Largest Component)** | 2 | 7 | 2 |
| **Average Shortest Path Length** | 1.86 | 3.07 | 1.86 |

---

## 3. Data Provenance & Zero-Leakage Verification

- **Curriculum Taxonomy**: Sourced strictly from static content metadata (`Info_Content.csv`). It contains zero student telemetry and zero temporal information.
- **Empirical Transition Matrix**: Fitted exclusively on `train.parquet` in lines 143–196 of `services/gnn_kt/src/graph_builder.py`. Zero validation or test student sequences are observed during graph construction.
- **Duplicate & Self-Loop Hygiene**:
  - `A_hier` contains 0 duplicate edges and 0 self-loops.
  - `A_trans` contains 0 duplicate edges and 0 self-loops (self-transitions c_t = c_{t+1} are strictly filtered at line 171).
  - `A_combined` adds exactly $N=150$ identity self-loops for spectral GCN/GAT stability.

Supporting data files exported to:
- [`graph_topology_metrics.json`](file:///d:/digital%20twin/services/gnn_kt/outputs/phase3/graph_topology_metrics.json)
- [`graph_edge_audit.csv`](file:///d:/digital%20twin/services/gnn_kt/outputs/phase3/graph_edge_audit.csv)
