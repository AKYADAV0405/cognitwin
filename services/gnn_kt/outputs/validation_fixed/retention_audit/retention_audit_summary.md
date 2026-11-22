# Phase 2 Retention Target & Temporal Leakage Forensic Audit

- **Total Verified Test Retention Pairs**: 1,464
- **Zero Same-Concept Leakage Compliance**: 100% (VERIFIED)
- **15-Minute Grid Timestamp Quantization**: 100.0%

## Intervening Practice Disentanglement

1. **Isolated Retention (0 Intervening Practice)**: 394 pairs
2. **Related-Concept Interference (Graph Neighbors)**: 903 pairs
3. **Unrelated-Concept Interference (Unconnected)**: 167 pairs

## Stratified Metric Performance

| Stratum                                         |   Sample Count |   Positive Rate | Retention AUC   |   Brier Score | Macro-F1   |
|:------------------------------------------------|---------------:|----------------:|:----------------|--------------:|:-----------|
| Overall Test Retention                          |           1464 |          0.7944 | 0.5389          |        0.2143 | 0.4592     |
| Repeated Question Only                          |             11 |          0.9091 | 0.8             |        0.2038 | 0.45       |
| Different Question Only                         |           1453 |          0.7935 | 0.5384          |        0.2144 | 0.4592     |
| Group A: Isolated Retention (0 Intervening)     |            394 |          0.698  | 0.5473          |        0.2274 | 0.4227     |
| Group B: Related Interference (Graph Neighbors) |            903 |          0.845  | 0.5188          |        0.2069 | 0.4627     |
| Group C: Unrelated Interference (No Graph Edge) |            167 |          0.7485 | 0.5074          |        0.2237 | 0.4924     |
| Delay <= 15 min (Quantized Immediate)           |            431 |          0.7169 | 0.5332          |        0.2272 | 0.4446     |
| Delay 15m - 60m (Within Session)                |             29 |          0.6552 | 0.4842          |        0.2346 | 0.4952     |
| Delay 1h - 24h (Same-Day Review)                |            195 |          0.8462 | 0.5954          |        0.2053 | 0.4583     |
| Delay 1d - 7d (Multi-Day Spaced Review)         |            427 |          0.8361 | 0.5018          |        0.2087 | 0.4547     |
| Delay > 7 days (Long-Term Retention)            |            382 |          0.8194 | 0.4941          |        0.209  | 0.4504     |
| <15m | isolated                                 |            235 |          0.6723 | 0.5012          |        0.2348 | 0.4057     |
| <15m | related_interference                     |            161 |          0.7888 | 0.5505          |        0.2157 | 0.4524     |
| <15m | unrelated_interference                   |             35 |          0.6857 | 0.6061          |        0.2297 | 0.6182     |
| 15m-60m | isolated                              |             17 |          0.5294 | 0.5139          |        0.2525 | 0.4711     |
| 15m-60m | related_interference                  |              8 |          0.75   | 0.3333          |        0.2274 | 0.4286     |
| 15m-60m | unrelated_interference                |              4 |          1      | N/A             |        0.1725 | N/A        |
| 1h-24h | isolated                               |             58 |          0.8103 | 0.6847          |        0.202  | 0.4476     |
| 1h-24h | related_interference                   |            108 |          0.8611 | 0.5254          |        0.2064 | 0.4627     |
| 1h-24h | unrelated_interference                 |             29 |          0.8621 | 0.63            |        0.2075 | 0.463      |
| 1d-7d | isolated                                |             67 |          0.6866 | 0.5518          |        0.2256 | 0.4071     |
| 1d-7d | related_interference                    |            298 |          0.8893 | 0.5034          |        0.2011 | 0.4707     |
| 1d-7d | unrelated_interference                  |             62 |          0.7419 | 0.4389          |        0.2273 | 0.4206     |
| >7d | isolated                                  |             17 |          0.8824 | 0.4333          |        0.1933 | 0.4688     |
| >7d | related_interference                      |            328 |          0.8293 | 0.4964          |        0.2074 | 0.4533     |
| >7d | unrelated_interference                    |             37 |          0.7027 | 0.4301          |        0.23   | 0.4127     |
