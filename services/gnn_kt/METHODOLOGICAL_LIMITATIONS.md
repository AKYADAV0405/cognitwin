# TPA-GKT Methodological Limitations & Experimental Protocols

> **Single Source of Truth for Training State Continuity & Experimental Protocol Seams**  
> *Branch: `research/tpa-gkt-validation-fixes`*

---

## 1. Sequence Windowing & Learner-State Initialization Protocol

### 1.1 Training Protocol: Independent Window Initialization
- In the current implementation, model training across all configurations (`DKT`, `SAKT`, `GKT`, `TPA-GKT`) uses **shuffled mini-batches** of interaction windows (length $\le 200$).
- Because interaction windows from different students are randomly shuffled to satisfy IID optimization assumptions, the recurrent state at the start of each training window is initialized independently:
  $$\text{initial\_state} = \mathbf{0} \quad (\texttt{gru\_state\_mode = "independent\_window\_reset"})$$
- **Academic Reporting Seam**: While cross-window next-response prediction targets are strictly preserved across boundaries (the final step of window $w$ predicts step 0 of window $w+1$), **the model is NOT trained with continuous cross-window truncated backpropagation through time (TBPTT)**. Any publication manuscript must explicitly report that training was conducted under independent window resets.

### 1.2 Evaluation Protocol: Documented State Modes
- For evaluation over long student interaction sequences, two distinct modes are available and must be explicitly reported:
  1. `independent_window_reset`: Each window is evaluated starting from $h_0 = \mathbf{0}$, consistent with the training distribution.
  2. `continuous_carryover`: Consecutive windows for the same student are processed in chronological order, with the detached final hidden state $h_{\text{final}}$ from window $w$ passed as $\text{initial\_state}$ into window $w+1$.
- Every experiment log, evaluation summary JSON, and exported table must record the exact `gru_state_mode` (`"independent_window_reset"` vs `"continuous_carryover"`).

---

## 2. SAKT Attention Masking & PyTorch Deprecation Resolution

1. **Boolean Key Padding Mask & Causal Mask Alignment**:
   - In PyTorch's `nn.MultiheadAttention`, passing mismatched mask types (e.g., FloatTensor `attn_mask` with $-\infty$ and BoolTensor `key_padding_mask` with `True`) triggered 5 `UserWarning: Support for mismatched key_padding_mask and attn_mask is deprecated`.
   - **Resolution**: Converted `causal_mask` in SAKT to `torch.bool` (`torch.triu(..., diagonal=1)`), where `True` indicates masked/disallowed attention. Both masks are now uniformly boolean, guaranteeing optimized C++ kernel execution and eliminating all 5 PyTorch warnings.
2. **Padding Invariance & Attention Isolation**:
   - Padded positions are isolated by setting `key_padding_mask=~mask`. Attention scores to padded keys are strictly $-\infty$ (masked), preventing padded tokens from influencing valid representations.
   - Outputs at padded positions are explicitly zeroed out: `x = x * mask.unsqueeze(-1).float()`.

---

## 3. Retention Target Construction, Intervening Practice & Timestamp Quantization

### 3.1 15-Minute Grid Timestamp Quantization
- **Forensic Finding**: In the Junyi Academy dataset, **100.0% of interaction timestamps are quantized to multiples of 900 seconds (15 minutes)**.
- In the legacy $<1\text{h}$ delay bucket:
  - **93.7%** of retention pairs have an elapsed time of exactly $\Delta t \le 15\text{ min}$ (431 / 460 pairs).
  - Only **6.3%** have $\Delta t \in (15\text{m}, 60\text{m}]$ (29 / 460 pairs).
- **Methodological Implication**: Within-session retention checks at $\le 15\text{ min}$ measure immediate repetition rather than true memory decay. Retention models in educational data mining frequently exhibit lower AUC on immediate review ($0.7298$) than on multi-day spaced review ($0.7905$) because immediate repetitions are dominated by item-level difficulty rather than elapsed memory forgetting.

### 3.2 Intervening Practice Disentanglement
- **100% Zero Same-Concept Leakage**: Forensic verification confirmed that for all 1,464 test retention pairs, **zero intervening attempts occurred on the queried concept $c$** between source interaction $t_1$ and target interaction $t_2$.
- **Mutually Exclusive & Collectively Exhaustive Partition**:
  1. **Group A: Isolated Retention (0 Intervening Practice)**: 394 pairs (26.9%) — student performed no other practice whatsoever during delay $\Delta t$.
  2. **Group B: Related-Concept Interference (Graph Neighbors)**: 903 pairs (61.7%) — student practiced concepts connected by prerequisite or transition graph edges during delay $\Delta t$.
  3. **Group C: Unrelated-Concept Interference (Graph Unconnected)**: 167 pairs (11.4%) — student practiced unrelated concepts in the curriculum.
- **Reporting Requirement**: Research publications must stratify retention evaluation across these three groups to disentangle retroactive inhibition from time-based decay.
