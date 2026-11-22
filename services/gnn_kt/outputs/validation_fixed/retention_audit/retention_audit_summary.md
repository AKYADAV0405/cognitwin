# Phase 2 Retention Target & Temporal Leakage Forensic Audit

- **Total Verified Test Retention Pairs**: 1,464
- **Zero Same-Concept Leakage Compliance**: 100% (VERIFIED)
- **15-Minute Grid Timestamp Quantization**: 100.0%

## Intervening Practice Disentanglement

1. **Isolated Retention (0 Intervening Practice)**: 394 pairs
2. **Related-Concept Interference (Graph Neighbors)**: 903 pairs
3. **Unrelated-Concept Interference (Unconnected)**: 167 pairs

## Stratified Metric Performance

| Stratum                                         |   Sample Count |   Positive Rate | Retention AUC   | AUC 95% CI       |   Brier Score | Macro-F1   | Inference Reliability Notes                                               |
|:------------------------------------------------|---------------:|----------------:|:----------------|:-----------------|--------------:|:-----------|:--------------------------------------------------------------------------|
| Overall Test Retention                          |           1464 |          0.7944 | 0.7581          | [0.7291, 0.7853] |        0.1411 | 0.5977     | Sufficient statistical power (N >= 50)                                    |
| Repeated Question Only                          |             11 |          0.9091 | 0.3             | [0.0000, 0.6000] |        0.1828 | 0.3889     | Small sample size (N < 20); high variance, interpret with extreme caution |
| Different Question Only                         |           1453 |          0.7935 | 0.7605          | [0.7317, 0.7915] |        0.1408 | 0.5996     | Sufficient statistical power (N >= 50)                                    |
| Group A: Isolated Retention (0 Intervening)     |            394 |          0.698  | 0.7261          | [0.6784, 0.7787] |        0.1864 | 0.6206     | Sufficient statistical power (N >= 50)                                    |
| Group B: Related Interference (Graph Neighbors) |            903 |          0.845  | 0.7412          | [0.6910, 0.7859] |        0.1162 | 0.5584     | Sufficient statistical power (N >= 50)                                    |
| Group C: Unrelated Interference (No Graph Edge) |            167 |          0.7485 | 0.7312          | [0.6475, 0.8154] |        0.1688 | 0.528      | Sufficient statistical power (N >= 50)                                    |
| Delay <= 15 min (Quantized Immediate)           |            431 |          0.7169 | 0.7183          | [0.6664, 0.7684] |        0.1813 | 0.63       | Sufficient statistical power (N >= 50)                                    |
| Delay 15m - 60m (Within Session)                |             29 |          0.6552 | 0.8632          | [0.6824, 0.9803] |        0.1593 | 0.6853     | Moderate sample size (20 <= N < 50); wide confidence interval             |
| Delay 1h - 24h (Same-Day Review)                |            195 |          0.8462 | 0.7388          | [0.6404, 0.8243] |        0.1218 | 0.5584     | Sufficient statistical power (N >= 50)                                    |
| Delay 1d - 7d (Multi-Day Spaced Review)         |            427 |          0.8361 | 0.7465          | [0.6811, 0.7992] |        0.1237 | 0.5529     | Sufficient statistical power (N >= 50)                                    |
| Delay > 7 days (Long-Term Retention)            |            382 |          0.8194 | 0.7755          | [0.7139, 0.8337] |        0.1237 | 0.5145     | Sufficient statistical power (N >= 50)                                    |
| <15m | isolated                                 |            235 |          0.6723 | 0.6758          | [0.6060, 0.7437] |        0.2065 | 0.6034     | Sufficient statistical power (N >= 50)                                    |
| <15m | related_interference                     |            161 |          0.7888 | 0.7677          | [0.6861, 0.8523] |        0.141  | 0.6725     | Sufficient statistical power (N >= 50)                                    |
| <15m | unrelated_interference                   |             35 |          0.6857 | 0.7197          | [0.5319, 0.9050] |        0.1968 | 0.5949     | Moderate sample size (20 <= N < 50); wide confidence interval             |
| 15m-60m | isolated                              |             17 |          0.5294 | 0.7917          | [0.4857, 1.0000] |        0.2016 | 0.6886     | Small sample size (N < 20); high variance, interpret with extreme caution |
| 15m-60m | related_interference                  |              8 |          0.75   | 1.0             | N/A              |        0.1239 | 0.4286     | Small sample size (N < 20); high variance, interpret with extreme caution |
| 15m-60m | unrelated_interference                |              4 |          1      | N/A             | N/A              |        0.0502 | N/A        | Small sample size (N < 20); high variance, interpret with extreme caution |
| 1h-24h | isolated                               |             58 |          0.8103 | 0.7427          | [0.5416, 0.9161] |        0.1424 | 0.6712     | Sufficient statistical power (N >= 50)                                    |
| 1h-24h | related_interference                   |            108 |          0.8611 | 0.7283          | [0.5874, 0.8446] |        0.1138 | 0.46       | Sufficient statistical power (N >= 50)                                    |
| 1h-24h | unrelated_interference                 |             29 |          0.8621 | 0.79            | [0.5614, 0.9630] |        0.1103 | 0.4528     | Moderate sample size (20 <= N < 50); wide confidence interval             |
| 1d-7d | isolated                                |             67 |          0.6866 | 0.7692          | [0.6570, 0.8729] |        0.18   | 0.5895     | Sufficient statistical power (N >= 50)                                    |
| 1d-7d | related_interference                    |            298 |          0.8893 | 0.6758          | [0.5737, 0.7704] |        0.098  | 0.4941     | Sufficient statistical power (N >= 50)                                    |
| 1d-7d | unrelated_interference                  |             62 |          0.7419 | 0.6726          | [0.5252, 0.8043] |        0.1864 | 0.5231     | Sufficient statistical power (N >= 50)                                    |
| >7d | isolated                                  |             17 |          0.8824 | 0.9667          | [0.8667, 1.0000] |        0.0673 | 0.4688     | Small sample size (N < 20); high variance, interpret with extreme caution |
| >7d | related_interference                      |            328 |          0.8293 | 0.7647          | [0.6910, 0.8301] |        0.1212 | 0.521      | Sufficient statistical power (N >= 50)                                    |
| >7d | unrelated_interference                    |             37 |          0.7027 | 0.7692          | [0.5903, 0.9277] |        0.1714 | 0.4565     | Moderate sample size (20 <= N < 50); wide confidence interval             |
