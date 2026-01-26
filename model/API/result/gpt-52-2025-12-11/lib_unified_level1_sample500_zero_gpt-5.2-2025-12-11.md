# Evaluation Report
**Level**: 1
**Dataset**: lib_unified.csv
**Shot Type**: zero
**Sample Size**: 500
**Model**: gpt-5.2-2025-12-11

## Accuracy Metrics
| Metric | Value | Count |
|--------|-------|-------|
| **Top-1** | 58.00% | 290/500 |
| **Top-3** | 90.40% | 452/500 |
| **Top-5** | 95.00% | 475/500 |

## Weighted Metrics (Top-1)
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.5800 |
| **Precision (Weighted)** | 0.6517 |
| **Recall (Weighted)** | 0.5800 |
| **F1 Score (Weighted)** | 0.5812 |

## Latency
- **Avg Duration**: 10471.45 ms
- **Total Duration**: 5235.73 s

## Sample Results (First 10)
|   ddc_code | predicted_codes                     | title                                  |
|-----------:|:------------------------------------|:---------------------------------------|
|        500 | ['300', '500', '100', '600', '900'] | Anxiety                                |
|        600 | ['600', '300', '000', '500', 'DDC'] | Construction Management                |
|        300 | ['800', '300', '000', '600', '400'] | The future of writing                  |
|        700 | ['700', '300', '900', '800', '400'] | Tiger roars in again                   |
|        700 | ['700', '300', '800', '900', '100'] | Andover now a target                   |
|        300 | ['300', '600', '500', '900', '000'] | Weyerhaeuser 3rd-Quarter Earnings Rise |
|        300 | ['900', '300', '400', '100', '000'] | Curfew eased in tense Kathmandu        |
|        300 | ['300', '900', '600', '000', '500'] | Halliburton may shed KBR unit          |
|        700 | ['700', '300', '900', '800', '000'] | Upcoming Golf                          |
|        700 | ['700', '300', '900', '500', '800'] | Wildcat strike stuns Hens              |