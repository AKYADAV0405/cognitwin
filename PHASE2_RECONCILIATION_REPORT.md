# Phase 2 Reconciliation Audit Report

> **Comprehensive Research Methodology & Forensic Reconciliation**  
> **Model**: Temporal Prerequisite-Aware Graph Knowledge Tracing (TPA-GKT)  
> **Dataset**: Junyi Academy 2020 (Math Curriculum)  
> **Branch**: `research/tpa-gkt-validation-fixes` (Commit `4f29281`)  
> **Status**: Completed, Fully Reconciled, and Verified by Automated Unit Tests

---

## Executive Overview

This report provides the full forensic reconciliation requested for Phase 2 prior to advancing to Phase 3. It rigorously investigates:
1. The mathematical relationship between the local 1,464 test retention pairs and the 16,285 pairs from the full 5,000-student benchmark.
2. The root cause of the empirical AUC discrepancy (0.5389 chance vs. 0.7581 local trained vs. 0.7772 full benchmark).
3. The exact timestamp forensic accounting (15-minute grid quantization, $\Delta t = 0$ exclusions, and potential selection bias).
4. Graph construction provenance, data leakage checks, and taxonomy vs. prerequisite distinctions.
5. Statistical reporting across all strata with bootstrap 95% confidence intervals and observational phrasing.

All supporting deliverables have been exported to disk:
- [`RETENTION_PAIR_ACCOUNTING.csv`](file:///d:/digital%20twin/RETENTION_PAIR_ACCOUNTING.csv)
- [`TIMESTAMP_FORENSIC_ACCOUNTING.csv`](file:///d:/digital%20twin/TIMESTAMP_FORENSIC_ACCOUNTING.csv)
- [`RETENTION_METRIC_RECONCILIATION.csv`](file:///d:/digital%20twin/RETENTION_METRIC_RECONCILIATION.csv)
- [`services/gnn_kt/tests/test_phase2_retention.py`](file:///d:/digital%20twin/services/gnn_kt/tests/test_phase2_retention.py) (7/7 tests passing)

---

## 1. Task 1: Retention-Pair Count & Population Reconciliation

### 1.1 Empirical Cohort Overlap Verification via Actual Student IDs
- **Methodological Disclosure**: We performed an exhaustive ID-level audit by reconstructing the exact student sampling and splitting pipeline against the 29,854 eligible students from `data/raw/Log_Problem.csv` (`data/processed/eligible_students.json`).
- **Initial Sampling Step (Lines 219–223 of `preprocess.py`)**:
  - `rng = np.random.RandomState(42)`
  - Sampling 500 students without replacement produces a **strict subset (500 / 500, 100.0%)** of the 5,000 students sampled under the identical random state.
- **Split Permutation Step (Lines 372–382 of `preprocess.py`)**:
  - Before splitting into 70% Train, 15% Val, and 15% Test, `preprocess.py` executes:
    ```python
    unique_students = np.sort(df["uuid"].unique())
    split_rng = np.random.RandomState(42)
    split_rng.shuffle(unique_students)
    ```
  - **Crucial Mathematical Discovery**: Because `split_rng.shuffle()` operates on an array of length 500 in fast-dev versus length 5,000 in full, the pseudo-random permutation order diverges completely. Consequently:
    - **The 75 local test students are NOT a strict subset of the 750 full test students.**
- **Empirical ID Overlap Breakdown ([`COHORT_OVERLAP_ACCOUNTING.csv`](file:///d:/digital%20twin/COHORT_OVERLAP_ACCOUNTING.csv))**:
  - **Exact Intersection**: **11 students** (14.67% of local test cohort).
  - **Local Test Students in Full Train**: **57 students** (76.00%).
  - **Local Test Students in Full Val**: **7 students** (9.33%).
  - **Local-Only Test Students**: **64 students** (85.33%).
  - **Full-Only Test Students**: **739 students** (98.53%).
  - **Overlapping Test Interactions**: **1,633 interactions** (14.02% of the 11,646 local test rows).
  - **Local-Only Test Interactions**: **10,013 interactions** (85.98%).
  - **Overlapping Test Retention Pairs**: **299 pairs** (20.42% of the 1,464 local test retention pairs).
  - **Local-Only Test Retention Pairs**: **1,165 pairs** (79.58%).

This empirical finding directly and legitimately explains why local test retention AUC is **0.7581** while full benchmark test retention AUC is **0.7772**: the models were evaluated on different student populations, and the full benchmark was trained on 10× more data (3,500 students vs. 350 students).

### 1.2 Algorithm, Schema & Filtering Consistency
- **Algorithm Identity**: Both cohorts use the exact same function, `compute_delayed_retention(df)` in `services/gnn_kt/src/preprocess.py` (lines 62–126).
- **Same-Concept Definition**: In both runs, a candidate retention pair is defined as two consecutive interactions $(t_1, t_2)$ on the same concept $c$ (`uuid`, `ucid`) by user $u$.
- **Delay Filtering**: Both runs enforce $\Delta t = \tau_{t_2} - \tau_{t_1} > 0$. Zero-delay repetitions ($\Delta t = 0$) within the same 15-minute timestamp block are excluded from delayed retention in both datasets.
- **Schema Parity**: Both Parquet datasets share the identical 35-column feature schema (`uuid`, `ucid`, `upid`, `is_correct`, `timestamp_sec`, `delta_t`, `norm_log_sec`, `has_retention_target`, `retention_target`, `retention_delta_sec`, `retention_bucket_id`, `retention_bucket_name`, etc.).

### 1.3 Exact Mutually Exclusive Interaction-Record Accounting Table
To resolve all arithmetic ambiguities, the population base is formally defined as **Total Interaction Records** ($N_{\text{total}}$). Every single interaction row in the split receives exactly one mutually exclusive disposition:
$$\text{Total Interactions} = N_{\text{valid}} + N_{\text{zero\_delay}} + N_{\text{terminal\_tail}} + N_{\text{neg}} + N_{\text{invalid}}$$

| Cohort | Total Students | Total Interaction Records | Valid Delayed Retention ($\Delta t > 0$) | Intra-Session Zero Delay ($\Delta t = 0$) | Negative Time Discrepancies ($\Delta t < 0$) | Missing / Invalid Timestamps ($t \le 0$) | Terminal Concept Tail (No Future Attempt) | Accounting Sum Check | Balances Exactly? | Retention Ratio (%) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| **Fast-Dev Test Split (Local)** | **75** | **11,646** | **1,464** | **8,998** | **0** | **0** | **1,184** | **11,646** | **YES (100.0%)** | **12.57%** |
| **Full Benchmark Test Split (Colab)** | **750** | **120,271** | **16,285** | **91,695** | **0** | **0** | **12,291** | **120,271** | **YES (100.0%)** | **13.54%** |
| **Combined Total** | **825** | **131,917** | **17,749** | **100,693** | **0** | **0** | **13,475** | **131,917** | **YES (100.0%)** | **13.45%** |

*Note on Terminal Concept Tails*: For every unique student-concept sequence $(u, c)$, the final interaction chronologically has no subsequent interaction on concept $c$. In the local test split, $N_{\text{terminal\_tail}} = 1,184$ matches mathematically to the exact number of unique $(u, c)$ groups in `test.parquet` (`df.groupby(["uuid", "ucid"]).ngroups == 1184`). All 5 categories are mutually exclusive and collectively exhaustive. Exported to disk at [`RETENTION_PAIR_ACCOUNTING.csv`](file:///d:/digital%20twin/RETENTION_PAIR_ACCOUNTING.csv).

---

## 2. Task 2: Investigation of the AUC Discrepancy

### 2.1 The Discrepancy Context
- Earlier reported full benchmark (5,000 students, Colab): **Retention AUC = 0.7772**.
- Fast-dev baseline table (`table3_results.csv` on local 500 students): **Retention AUC = 0.7581**.
- Initial Phase 2 forensic audit script output: **Retention AUC = 0.5389**.

### 2.2 Root-Cause Forensic Discovery: Silent State-Dict Unpacking Failure
Through code inspection of `services/gnn_kt/src/retention_audit.py` and PyTorch execution tracing, the exact technical cause was uncovered:
1. **Checkpoint Structure**: When checkpoints are saved in `services/gnn_kt/src/train.py`, the saved object is a composite dictionary:
   ```python
   checkpoint = {
       "epoch": epoch,
       "model_name": "tpa_gkt",
       "model_state_dict": model.state_dict(),  # <-- actual weights
       "config": config,
       "num_concepts": num_concepts,
       "num_problems": num_problems,
   }
   ```
2. **The Defect in the Initial Audit Script**:
   In the initial draft of `retention_audit.py` (lines 252–264):
   ```python
   state_dict = torch.load(checkpoint_path, map_location=device)
   model = TPAGKT(...)
   model.load_state_dict(state_dict, strict=False)  # <-- BUG
   ```
   Because `state_dict` was the outer wrapper dictionary (where the parameter tensors resided under the key `"model_state_dict"`), **zero keys matched the model parameters**.
3. **Silent Failure via `strict=False`**:
   Passing `strict=False` instructed PyTorch to silently ignore all missing keys. As a result, **no trained weights were loaded**. The model evaluated during the initial audit was a **completely uninitialized random neural network**.
4. **Validation Proof**:
   Evaluating an untrained random neural network on a binary retention target with 79.4% positive class imbalance yields a near-chance ROC-AUC of **0.5389**.

### 2.3 Correction & True Reconciled Metrics
When the checkpoint is loaded using the standard model constructor `build_model()` with pre-registered graph buffers and strict state-dict verification (`strict=True` on `checkpoint["model_state_dict"]`):
- **Aliged Predictions**: **1,464 of 1,464 pairs (100.0%)** aligned with 0 unmatched.
- **Reconciled Retention AUC on Local Test**: **0.7581** (95% CI: `[0.7291, 0.7853]`).
- **Reconciled Brier Score**: **0.1411**.
- **Reconciled Macro-F1**: **0.5977**.

This reconciles the numbers:
- Local 500-student development checkpoint: **AUC = 0.7581** (proven identical to `table3_results.csv`).
- Full 5,000-student benchmark checkpoint: **AUC = 0.7772** (higher due to 10× training data: 3,500 students vs. 350 students).
- Neither metric was manufactured; the 0.5389 figure was an artifact of uninitialized weights in the preliminary script.

---

## 3. Task 3: Timestamp Forensic Accounting & Selection Bias Analysis

### 3.1 15-Minute Grid Quantization Audit
Junyi Academy 2020 timestamps are recorded in Taiwan Standard Time (`timestamp_TW`) and quantized to **900-second (15-minute) intervals**:
- **Total Test Interactions**: 11,646.
- **Missing or Invalid Timestamps ($t \le 0$)**: **0 (0.00%)**.
- **Negative Timestamp Reversals ($\Delta t < 0$)**: **0 (0.00%)**.
- **Interactions Sharing Identical Timestamps with Another Interaction (Same Student)**: **11,474 (98.52%)**.

### 3.2 Candidate Transitions and the $\Delta t = 0$ Cluster
- Across all 75 test students, there are **10,462 candidate same-concept consecutive transitions** $(u, c, t_1) \rightarrow (u, c, t_2)$.
- **Transitions with $\Delta t == 0$**: **8,998 (86.01%)**.
- **Transitions with $\Delta t > 0$**: **1,464 (13.99%)**.

### 3.3 Selection Bias Investigation
Why do 86.01% of same-concept transitions have $\Delta t = 0$, and does filtering them introduce selection bias?
1. **Intra-Session Practice vs. Delayed Retention**:
   - Students in Junyi solve clusters of 5–15 problem instances for a given math concept during a single learning session.
   - Because all problems solved within a 15-minute sitting receive the identical quantized timestamp ($t_1 = t_2$), their elapsed time is $\Delta t = 0$.
   - By definition in cognitive psychology (Ebbinghaus forgetting curve), "delayed retention" measures performance across a non-zero time delay. Consecutive items attempted within the same sitting represent **immediate sequential practice**, which is explicitly modeled and evaluated by the **next-response prediction head** ($r_t \rightarrow r_{t+1}$).
2. **Empirical Performance Comparison**:
   - Target correctness for intra-session transitions ($\Delta t = 0$): **74.91%** (10,462 transitions).
   - Target correctness for delayed retention ($\Delta t > 0$): **79.44%** (1,464 pairs).
3. **Selection Bias Conclusion**:
   - Filtering $\Delta t = 0$ transitions selects students who returned to a concept after an elapsed interval ($\ge 15$ min, hours, or days).
   - The slightly higher accuracy (79.44% vs 74.91%) reflects the fact that students who return to review a concept have had more cumulative practice.
   - **Methodological Requirement for Paper 1**: The paper must explicitly disclose that the delayed-retention evaluation is defined over inter-session spaced retrieval ($\Delta t > 0$), while intra-session problem-solving is evaluated via next-response AUC.

---

## 4. Task 4: Graph Leakage Verification & Taxonomy vs. Prerequisite Distinction

### 4.1 Zero-Leakage Graph Provenance Verification
- **Verification Question**: Was the transition graph $A_{\text{trans}}$ constructed exclusively from training data, or did validation/test interactions leak into edge weights?
- **Forensic Verification in `services/gnn_kt/src/graph_builder.py`**:
  ```python
  # Line 257 & Line 293:
  train_parquet = data_proc_dir / "train.parquet"
  trans_edge_index, trans_edge_weight, trans_dict = build_transition_graph(
      train_parquet,
      num_nodes=num_nodes,
      transition_threshold=trans_threshold,
      min_transition_count=min_trans_count,
  )
  ```
- **Finding**: $A_{\text{trans}}$ is constructed **strictly from `train.parquet`**. Neither validation nor test interactions are accessed during edge frequency counting or transition thresholding. Zero data leakage is verified.

### 4.2 Distinguishing Curriculum Taxonomy from Causal Prerequisites
- **Curriculum Taxonomy Graph ($A_{\text{hier}}$)**:
  - Constructed from `Info_Content.csv` using hierarchical topic metadata (`level1_id`, `level2_id`, `level3_id`, `level4_id`).
  - Represents **co-membership in the curriculum taxonomy** (e.g., concepts sharing the unit "Quadratic Equations").
  - Bidirectional, symmetric clustering proxy.
- **Empirical Transition Graph ($A_{\text{trans}}$)**:
  - Constructed from empirical interaction frequencies $P(c_{t+1} = j \mid c_t = i) \ge 0.05$ on training sequences.
  - Represents common student navigational pathways through the platform.
- **Academic Precision for Paper 1**:
  - Neither graph is an expert-curated causal prerequisite DAG (which would require directed acyclic constraints where mastery of $A$ is strictly necessary to unlock $B$).
  - In Paper 1, the graph must be described as a **"prerequisite-informed curriculum graph"** or **"structural knowledge topology"**, avoiding unwarranted causal claims about prerequisite necessity.

---

## 5. Task 5: Statistical & Scientific Reporting

The table below presents the reconciled test metrics using exact interaction-ID alignment (`src_orig_idx`), evaluated on the trained checkpoint:

| Stratum Description | Sample Count | % of Valid | Positive Rate | Retention AUC | Bootstrap 95% CI | Brier Score | Macro-F1 | Sample Size & Reliability Notes |
|:---|---:|---:|---:|---:|:---:|---:|---:|:---|
| **Overall Test Retention** | **1,464** | **100.0%** | **79.44%** | **0.7581** | **[0.7291, 0.7853]** | **0.1411** | **0.5977** | Sufficient statistical power ($N \ge 50$) |
| *Question Repeatedness* | | | | | | | | |
| ├── Different Question Only ($p_1 \ne p_2$) | 1,453 | 99.25% | 79.35% | 0.7605 | [0.7317, 0.7915] | 0.1408 | 0.5996 | Sufficient statistical power ($N \ge 50$) |
| └── Repeated Question Only ($p_1 = p_2$) | 11 | 0.75% | 90.91% | 0.3000 | [0.0000, 0.6000] | 0.1828 | 0.3889 | Small sample size ($N=11$); high variance, interpret with extreme caution |
| *Intervening Practice Category* | | | | | | | | |
| ├── Group A: Isolated Retention (0 Intervening) | 394 | 26.91% | 69.80% | 0.7261 | [0.6784, 0.7787] | 0.1864 | 0.6206 | Sufficient statistical power ($N \ge 50$) |
| ├── Group B: Related Interference (Graph Neighbors) | 903 | 61.68% | 84.50% | 0.7412 | [0.6910, 0.7859] | 0.1162 | 0.5584 | Sufficient statistical power ($N \ge 50$) |
| └── Group C: Unrelated Interference (No Graph Edge) | 167 | 11.41% | 74.85% | 0.7312 | [0.6475, 0.8154] | 0.1688 | 0.5280 | Sufficient statistical power ($N \ge 50$) |
| *Fine-Grained Delay Intervals* | | | | | | | | |
| ├── Delay $\le 15$ min (Quantized Immediate) | 431 | 29.44% | 71.69% | 0.7183 | [0.6664, 0.7684] | 0.1813 | 0.6300 | Sufficient statistical power ($N \ge 50$) |
| ├── Delay $15$m -- $60$m (Within Session) | 29 | 1.98% | 65.52% | 0.8632 | [0.6824, 0.9803] | 0.1593 | 0.6853 | Moderate sample size ($N=29$); wide confidence interval |
| ├── Delay $1$h -- $24$h (Same-Day Review) | 195 | 13.32% | 84.62% | 0.7388 | [0.6404, 0.8243] | 0.1218 | 0.5584 | Sufficient statistical power ($N \ge 50$) |
| ├── Delay $1$d -- $7$d (Multi-Day Spaced Review) | 427 | 29.17% | 83.61% | 0.7465 | [0.6811, 0.7992] | 0.1237 | 0.5529 | Sufficient statistical power ($N \ge 50$) |
| └── Delay $> 7$ days (Long-Term Retention) | 382 | 26.09% | 81.94% | 0.7755 | [0.7139, 0.8337] | 0.1237 | 0.5145 | Sufficient statistical power ($N \ge 50$) |

### 5.1 Observational Scientific Phrasing (Replacing Causal Claims)
- **Previous statement**: *"Practicing topological neighbors acts as a positive cognitive anchor that mitigates forgetting."*
- **Reconciled observational statement**: *"In observational telemetry, students who engaged in practice on curriculum-adjacent concepts (Group B) exhibited higher empirical retention accuracy (84.50%) than students with no intervening activity (69.80%, Group A) or students practicing unrelated concepts (74.85%, Group C). While this pattern is consistent with the hypothesis that topic proximity reinforces shared retrieval pathways, non-experimental observational data cannot establish a causal anchoring mechanism, as higher-performing students may also be more likely to engage in deliberate curriculum exploration."*

---

## 6. Verification and Automated Test Results

The test suite in `services/gnn_kt/tests/test_phase2_retention.py` was expanded with 2 additional tests:
1. `test_retention_reconciliation_csv_deliverables`: Validates file existence, row counts, and numerical invariants across all 3 reconciliation CSVs.
2. `test_graph_zero_leakage_provenance`: Verifies that graph files have valid edge dimensions, zero NaNs, bounded edge weights, and match the train split.

Full pytest execution across the entire repository:
```bash
services\gnn_kt\tests\test_cold_start.py ....                            [  8%]
services\gnn_kt\tests\test_models.py ...................                 [ 48%]
services\gnn_kt\tests\test_phase1_pipeline.py ........                   [ 65%]
services\gnn_kt\tests\test_phase2_retention.py .......                   [ 80%]
services\gnn_kt\tests\test_preprocess_and_graph.py .......               [ 93%]
services\gnn_kt\tests\test_train_eval.py ....                            [100%]

============================= 49 passed in 34.12s =============================
```
- **Total Tests**: 49
- **Passed**: 49 (100% pass rate)
- **Warnings / Failures**: 0

---

## 7. Next Steps & Approval Gate

The Phase 2 reconciliation audit is complete. All 5 tasks are fully resolved with transparent mathematical proofs, code-level bug diagnostics, and exported CSV artifacts.

**Phase 3 (Graph Architecture & Ablation Validation) remains blocked awaiting your review and explicit approval.**
