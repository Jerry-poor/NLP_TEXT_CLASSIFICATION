# Evaluation Report
**Level**: 1
**Dataset**: lib_unified.csv
**Shot Type**: zero
**Sample Size**: 500
**Model**: deepseek-chat

## Accuracy Metrics
| Metric | Value | Count |
|--------|-------|-------|
| **Top-1** | 62.20% | 311/500 |
| **Top-3** | 91.40% | 457/500 |
| **Top-5** | 96.60% | 483/500 |

## Weighted Metrics (Top-1)
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.6220 |
| **Precision (Weighted)** | 0.6363 |
| **Recall (Weighted)** | 0.6220 |
| **F1 Score (Weighted)** | 0.6196 |

## Latency
- **Avg Duration**: 11704.66 ms
- **Total Duration**: 5852.33 s

## Sample Results (First 10)
|   ddc_code | predicted_codes                     | title                                                           |
|-----------:|:------------------------------------|:----------------------------------------------------------------|
|        700 | ['700', '300', '100', '400', '800'] | Andover now a target                                            |
|        700 | ['300', '900', '100', '200', '400'] | Accuser Told Bryant 'No'                                        |
|        300 | ['400', '300', '700', '800', '600'] | The future of writing                                           |
|        700 | ['300', '700', '900', '400', '500'] | Wildcat strike stuns Hens                                       |
|        000 | ['000', '600', '500', '700', '900'] | Data structures                                                 |
|        300 | ['300', '600', '900', '500', '400'] | Auto-body chain settles fraud suit                              |
|        300 | ['300', '100', '500', '600', '400'] | Future of law and economics                                     |
|        600 | ['600', '300', '000', '500', '700'] | Construction Management                                         |
|        500 | ['600', '000', '700', '500', '300'] | Media 100 HD video editing system released for Mac (MacCentral) |
|        300 | ['300', '600', '500', '000', '900'] | Investors in StarHub sell shares                                |