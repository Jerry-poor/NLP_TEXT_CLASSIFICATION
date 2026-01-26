# Evaluation Report
**Level**: 1
**Dataset**: lib_unified.csv
**Shot Type**: zero
**Sample Size**: 500
**Model**: qwen3-235b-a22b-instruct-2507

## Accuracy Metrics
| Metric | Value | Count |
|--------|-------|-------|
| **Top-1** | 58.60% | 293/500 |
| **Top-3** | 88.60% | 443/500 |
| **Top-5** | 93.00% | 465/500 |

## Weighted Metrics (Top-1)
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.5860 |
| **Precision (Weighted)** | 0.6128 |
| **Recall (Weighted)** | 0.5860 |
| **F1 Score (Weighted)** | 0.5871 |

## Latency
- **Avg Duration**: 11079.98 ms
- **Total Duration**: 5539.99 s

## Sample Results (First 10)
|   ddc_code | predicted_codes                     | title                                       |
|-----------:|:------------------------------------|:--------------------------------------------|
|        100 | ['300', '500', '600', '900']        | Attention                                   |
|        700 | ['700', '300', '900', '400', '800'] | Tiger roars in again                        |
|        600 | ['600', '300', '000', '500', '900'] | Construction Management                     |
|        300 | ['300', '600', '500', '000', '900'] | Weyerhaeuser 3rd-Quarter Earnings Rise      |
|        300 | ['900', '300', '100', '200', '700'] | Children Return to Classes in Russia        |
|        700 | ['300', '900', '700', '400', 'DDC'] | Minnesota Timberwolves Center Arrested (AP) |
|        300 | ['300', '600', '900']               | Retail #39;s Little Guys Come Back          |
|        700 | ['700', '300', '400', '800', '900'] | Andover now a target                        |
|        300 | ['900', '300', '400', 'DDC', '700'] | Curfew eased in tense Kathmandu             |
|        300 | ['300', '000', '600', '900', '400'] | Auto-body chain settles fraud suit          |