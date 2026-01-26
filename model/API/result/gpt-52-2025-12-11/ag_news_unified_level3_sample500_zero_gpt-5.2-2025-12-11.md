# Evaluation Report
**Level**: 3
**Dataset**: ag_news_unified.csv
**Shot Type**: zero
**Sample Size**: 500
**Model**: gpt-5.2-2025-12-11

## Accuracy Metrics
| Metric | Value | Count |
|--------|-------|-------|
| **Top-1** | 52.00% | 260/500 |
| **Top-3** | 74.60% | 373/500 |
| **Top-5** | 79.80% | 399/500 |

## Weighted Metrics (Top-1)
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.5200 |
| **Precision (Weighted)** | 0.8208 |
| **Recall (Weighted)** | 0.5200 |
| **F1 Score (Weighted)** | 0.5542 |

## Latency
- **Avg Duration**: 10564.97 ms
- **Total Duration**: 5282.48 s

## Sample Results (First 10)
|   ddc_code | predicted_codes                     | title                                                       |
|-----------:|:------------------------------------|:------------------------------------------------------------|
|        500 | ['505', '506', '502', '507', '509'] | Sony Shows Smaller PlayStation 2 (AP)                       |
|        332 | ['330', '336']                      | Dollar Drops Further as Central Banks Reassess Reserves     |
|        796 | ['796', '790', '791', '792', '794'] | Report: Spurrier Will Take S. Carolina Job (AP)             |
|        796 | ['796', '790', '791', '794', '792'] | Real Back on Track                                          |
|        796 | ['796', '790', '791', '794', '792'] | Warner to Start for Giants This Week (AP)                   |
|        500 | ['506', '507', '509', '505', '500'] | Corning begins work on Taiwan LCD facility                  |
|        796 | ['796', '790', '791', '792', '794'] | Ohio State #39;s big plays kill Wolverines                  |
|        330 | ['330', '338']                      | Rapidly expanding Vietnam Airlines ready to take on America |
|        320 | ['320', '324', '323', '322', '321'] | Homeless families total 100,000                             |
|        330 | ['330', '336']                      | Congressman Spratt wants Fed to                             |