# Temporal Prerequisite-Aware Graph Knowledge Tracing (TPA-GKT)
## Paper 1 Specification & System Architecture

> **Dataset**: Junyi Academy 2020 (`data/raw/Info_Content.csv`, `data/raw/Info_UserData.csv`, `data/raw/Log_Problem.csv`)  
> **Target Scope**: Graph-based Knowledge Tracing modeling dynamic concept mastery, forgetting risk, and prerequisite propagation under dual-signal telemetry.

---

## 1. Mathematical Formulation

Knowledge Tracing (KT) models a learner's evolving concept mastery over sequential problem interactions:
$$X_u = \left( x_1^u, x_2^u, \dots, x_{T_u}^u \right)$$

where each interaction tuple at time step $t$ is denoted as:
$$x_t^u = \left( c_t, y_t, \Delta t_t, r_t, h_t, a_t \right)$$

- $c_t \in \{1, \dots, N_c\}$: Target concept / knowledge component index.
- $y_t \in \{0, 1\}$: Binary correctness label (active quiz or problem attempt).
- $\Delta t_t = t_t - t_{t-1}$: Elapsed time since previous student interaction (modeling forgetting decay).
- $r_t \in \mathbb{R}^+$: Response time in seconds.
- $h_t \in \mathbb{N}$: Number of hints requested.
- $a_t \in \mathbb{N}$: Number of attempts before arriving at an answer.

### Objective
At step $t+1$, given interaction history $X_{1:t}^u$ and the combined prerequisite knowledge graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{A})$, jointly predict:
1. **Response Correctness Head**: $P(y_{t+1} = 1 \mid c_{t+1}, X_{1:t}^u, \mathcal{G}) \in [0, 1]$
2. **Concept Node Mastery Head**: $\mathbf{m}_t \in [0, 1]^{N_c}$ across all concepts in the knowledge graph.
3. **Retention / Forgetting Decay Head**: $\mathbf{R}_t = \exp\left(-\frac{\Delta t}{\mathbf{S}_t}\right)$, where $\mathbf{S}_t$ represents memory stability.

---

## 2. 5-Phase Technical Pipeline

```
  Phase 1: Chunked Preprocessing & Anti-Leakage Split (70/15/15)
                             │
                             ▼
  Phase 2: Dual Knowledge Graph Construction (A_hier, A_trans, A_comb)
                             │
                             ▼
  Phase 3: 4-Stage TPA-GKT Model Architecture (3 Heads + 6 Ablations)
                             │
                             ▼
  Phase 4: Competitive Baselines (DKT, SAKT, GKT)
                             │
                             ▼
  Phase 5: Multi-Seed Training, Bucketed Retention Eval & ECE Calibration
```

---

### Phase 1: Chunked Preprocessing & Anti-Leakage Protocol

1. **Large-File Chunked Streaming**:
   - `Log_Problem.csv` is approximately 2.95 GB with millions of interaction records.
   - Streaming reader uses `pandas.read_csv(..., chunksize=500000)` into Apache Parquet chunks (`data/processed/interactions_chunk_*.parquet`).
   - Memory consumption strictly bounded to $< 2.0\text{ GB}$ RAM throughout ingestion.

2. **Student & Concept Filtering**:
   - Target Top-$N_c$ concepts ($N_c = 150$) by total interaction frequency to ensure dense knowledge graph connectivity.
   - Filter students with interaction length $L \in [\text{min\_interactions}, \text{max\_interactions}]$ (e.g., $30 \le L \le 1000$).
   - Sample up to `max_students` (e.g., 5,000 for standard run; 500 for `--fast-dev-run`).

3. **Strict Zero-Leakage Split Protocol (`student_70_15_15`)**:
   - Partition students disjointly into **Train (70%)**, **Validation (15%)**, and **Test (15%)**.
   - **Crucial Rule**: Concept vocabularies, response time min/max scalers, transition graphs, and prior statistics must be computed **strictly on the Train split**.
   - Validation and Test splits are transformed using fitted Train statistics; unseen values map to `<UNK>`.

---

### Phase 2: Dual Knowledge Graph Construction (Structural & Behavioral Prerequisite Proxies)

> [!NOTE] Academic Terminology & Structural Scope
> In naturalistic educational datasets like Junyi Academy, an explicit ground-truth prerequisite DAG is not directly provided. Instead, we operationalize two principled proxies:
> 1. $\mathbf{A}_{\text{hier}}$ is the **Curriculum Hierarchy Graph** (capturing pedagogical and thematic clustering from curriculum taxonomy levels).
> 2. $\mathbf{A}_{\text{trans}}$ is the **Empirical Behavioral Transition Graph** (capturing learner traversal trajectories from interaction sequences).
> Together, they serve as dual structural and empirical proxies for knowledge component dependency relationships.

1. **Curriculum Hierarchy Graph ($\mathbf{A}_{\text{hier}}$)**:
   - Derived from `Info_Content.csv` structure (`level1_id`, `level2_id`, `level3_id`, `level4_id`, `difficulty`).
   - Concepts sharing a common parent topic receive undirected taxonomic co-topic edges.
   - Serves as a static structural proxy for curricular closeness.

2. **Empirical Behavioral Transition Graph ($\mathbf{A}_{\text{trans}}$)**:
   - Computed strictly across training student interaction sequences (zero leakage):
     $$C_{i, j} = \sum_{u \in \text{Train}} \sum_{t=1}^{T_u - 1} \mathbb{I}\left(c_t^u = i \wedge c_{t+1}^u = j\right)$$
   - Transition probability:
     $$P(j \mid i) = \frac{C_{i, j}}{\sum_k C_{i, k}}$$
   - Thresholding: Keep edge $(i, j)$ if $C_{i, j} \ge \text{min\_transition\_count}$ (e.g., 10) and $P(j \mid i) \ge \text{transition\_threshold}$ (e.g., 0.05).
   - Serves as a behavioral empirical proxy for learner progression paths.

3. **Unified Combined Adjacency ($\mathbf{A}_{\text{combined}}$)**:
   $$\mathbf{A}_{\text{combined}} = \alpha \mathbf{A}_{\text{hier}} + (1 - \alpha) \mathbf{A}_{\text{trans}}$$
   Normalized via symmetric normalized Laplacian: $\mathbf{\hat{A}} = \mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{\tilde{A}} \mathbf{\tilde{D}}^{-\frac{1}{2}}$, where $\mathbf{\tilde{A}} = \mathbf{A}_{\text{combined}} + \mathbf{I}$.

---

### Phase 3: 4-Stage TPA-GKT Architecture & Heads

#### Stage 1: Multimodal Interaction Embedding
Each interaction $x_t$ is embedded into dimension $d_{\text{embed}}$:
$$\mathbf{e}_t = \mathbf{E}_{\text{concept}}(c_t) + \mathbf{E}_{\text{correct}}(y_t) + \mathbf{W}_r \phi(r_t) + \mathbf{W}_h \phi(h_t) + \mathbf{W}_a \phi(a_t) + \mathbf{W}_{\Delta t} \psi(\Delta t_t)$$
where $\phi(\cdot)$ is log-transformed continuous feature and $\psi(\Delta t) = \log(1 + \Delta t)$ represents elapsed study gap.

#### Stage 2: Dual-Graph Message Passing
Propagate prerequisite dependencies and mastery signals across neighboring concepts:
$$\mathbf{H}^{(l+1)} = \sigma\left( \mathbf{\hat{A}} \mathbf{H}^{(l)} \mathbf{W}_{\text{gnn}}^{(l)} \right)$$
Supported layer types: **GCN** (Kipf & Welling) and **GAT** (Veličković et al. multi-head attention).

#### Stage 3: Temporal Recurrent Knowledge Tracing
Track student state trajectory with an LSTM / GRU recurrent cell:
$$\mathbf{h}_t^u = \text{LSTMCell}\left( [\mathbf{e}_t, \mathbf{z}_{c_t}^{(L)}], \mathbf{h}_{t-1}^u \right)$$
where $\mathbf{z}_{c_t}^{(L)}$ is the graph-propagated node representation for target concept $c_t$.

#### Stage 4: Multi-Task Prediction Heads
1. **Next-Response Correctness Head**:
   $$\hat{y}_{t+1} = \sigma\left( \mathbf{W}_y [\mathbf{h}_t^u \,\|\, \mathbf{g}_{c_{t+1}} \,\|\, \mathbf{e}_{q_{t+1}}] + b_y \right) \in [0, 1]$$
   Evaluates student performance on specific problem $q_{t+1}$ mapped to knowledge component $c_{t+1}$.

2. **Concept Node Mastery Head**:
   $$\hat{m}_{u,c,t} = \sigma\left( \mathbf{W}_m [\mathbf{h}_t^u \,\|\, \mathbf{g}_c] + b_m \right) \in [0, 1]$$
   Computes estimated mastery for all $N_c$ concepts simultaneously.
   *Supervision & Semantic Scope*: In non-synthetic educational telemetry (such as Junyi Academy), ground-truth mastery is an unobserved latent cognitive construct. During training, the Mastery Head is supervised via an auxiliary concept-level masked binary cross-entropy loss $\lambda_M \mathcal{L}_M$ ($\lambda_M = 0.2$) on $y_{t+1}$ whenever concept $c_{t+1} = c$. The resulting score represents an *operationally calibrated latent mastery belief* rather than an independently verified psychometric ground truth.

3. **Time-Conditioned Delayed-Retention Head**:
   $$\hat{R}_{u,c,t,\delta} = \sigma\left( \mathbf{W}_r [\mathbf{h}_t^u \,\|\, \mathbf{g}_c \,\|\, \log(1+\delta) \,\|\, \mathbf{v}_{u,c,t}] + b_r \right) \in [0, 1]$$
   Predicts memory retention probability across continuous delay $\delta$, conditioned on historical practice summary $\mathbf{v}_{u,c,t}$ strictly prior to time $t$. Total training objective:
   $$\mathcal{L} = \mathcal{L}_R + \lambda_M \mathcal{L}_M + \lambda_F \mathcal{L}_F$$

#### 6 Ablation Flags
1. `w/o A_hier`: Zeroes out curriculum hierarchy edges ($\alpha = 0$).
2. `w/o A_trans`: Zeroes out empirical transition edges ($\alpha = 1$).
3. `w/o Time_Decay`: Sets $\psi(\Delta t) = 0$ and removes elapsed-time inputs.
4. `w/o Attempt_Hint_Features`: Restricts input embedding to purely `(concept, correctness)`.
5. `w/o Graph_Propagation`: Bypasses Stage 2 GNN message passing (pure recurrent KT).
6. `GCN vs GAT`: Toggles spectral GCN convolution vs learned attentional aggregation.

---

### Phase 4: Competitive Baselines

For rigorous academic benchmarking against identical train/val/test splits:
1. **DKT (Deep Knowledge Tracing, Piech et al. 2015)**: Standard LSTM mapping concatenated `(concept, correctness)` embeddings to next-step prediction.
2. **SAKT (Self-Attentive Knowledge Tracing, Pandey & Karypis 2019)**: Multi-head transformer self-attention over historical exercise sequences.
3. **GKT (Graph Knowledge Tracing, Nakagawa et al. 2019)**: Standard knowledge tracing with concept transition graph propagation.

---

### Phase 5: Multi-Seed Training, Bucketed Retention & Calibration

1. **Multi-Seed Protocol**:
   - Every experiment is evaluated across 3 distinct seeds: `[42, 100, 2026]`.
   - Results reported as $\text{Mean} \pm \text{Std}$.

2. **Evaluation Metrics**:
   - **AUC (ROC)**: Primary ranking metric for correctness prediction.
   - **RMSE / Brier Score**: Calibration and probabilistic error metric.
   - **Accuracy**: Binary thresholded correctness metric.

3. **Bucketed Retention Evaluation (Paper Section IV.E)**:
   - To empirically prove the model captures forgetting curves over time gaps, evaluate test AUC/RMSE segmented into 4 interaction delay buckets:
     - Bucket 1: $\Delta t < 1 \text{ day}$ (Immediate practice)
     - Bucket 2: $1 \le \Delta t < 7 \text{ days}$ (Short-term review)
     - Bucket 3: $7 \le \Delta t < 30 \text{ days}$ (Medium-term retention)
     - Bucket 4: $\Delta t \ge 30 \text{ days}$ (Long-term retention / high forgetting risk)

4. **Expected Calibration Error (ECE)**:
   - Segment predicted probabilities into $M=10$ bins to evaluate whether confidence matches observed accuracy.

5. **Fast Development Run (`--fast-dev-run`)**:
   - Downsamples dataset to 500 students and 50 concepts, training for 2 epochs to enable rapid end-to-end debugging and CI verification in $< 60$ seconds.
