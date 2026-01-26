# Evaluation Report
**Level**: 3
**Dataset**: ag_news_unified.csv
**Shot Type**: zero
**Sample Size**: 500
**Model**: qwen3-235b-a22b-instruct-2507

## Accuracy Metrics
| Metric | Value | Count |
|--------|-------|-------|
| **Top-1** | 47.40% | 237/500 |
| **Top-3** | 60.00% | 300/500 |
| **Top-5** | 65.20% | 326/500 |

## Weighted Metrics (Top-1)
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.4740 |
| **Precision (Weighted)** | 0.8220 |
| **Recall (Weighted)** | 0.4740 |
| **F1 Score (Weighted)** | 0.4885 |

## Latency
- **Avg Duration**: 10972.01 ms
- **Total Duration**: 5486.01 s

## Sample Results (First 10)
|   ddc_code | predicted_codes                     | title                                                          |
|-----------:|:------------------------------------|:---------------------------------------------------------------|
|        330 | ['330', '336']                      | China vows currency shift but mum on date                      |
|        330 | ['330', '336']                      | Congressman Spratt wants Fed to                                |
|        332 | ['330', '336']                      | Dollar Drops Further as Central Banks Reassess Reserves        |
|        330 | ['334', '336', '330', '335']        | Closing the giving gap                                         |
|        796 | ['796', '791', '790', '792', '793'] | Report: Spurrier Will Take S. Carolina Job (AP)                |
|        330 | ['330', '338', '336']               | UPDATE 2-UK #39;s Linx drops ITW offer after Danaher steams in |
|        500 | ['502', '505', '506', '507', '509'] | Dell unveils holiday lineup, including new plasma TVs          |
|        500 | ['502', '506', '505', '509', '507'] | Sony Shows Smaller PlayStation 2 (AP)                          |
|        330 | ['330', '338']                      | Rapidly expanding Vietnam Airlines ready to take on America    |
|        330 | ['338', '330', '336']               | AOL's Viral Marketing                                          |