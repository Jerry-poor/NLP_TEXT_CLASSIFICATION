# Evaluation Report
**Level**: 3
**Dataset**: ag_news_unified.csv
**Shot Type**: zero
**Sample Size**: 500
**Model**: deepseek-chat

## Accuracy Metrics
| Metric | Value | Count |
|--------|-------|-------|
| **Top-1** | 48.20% | 241/500 |
| **Top-3** | 62.20% | 311/500 |
| **Top-5** | 73.80% | 369/500 |

## Weighted Metrics (Top-1)
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.4820 |
| **Precision (Weighted)** | 0.8195 |
| **Recall (Weighted)** | 0.4820 |
| **F1 Score (Weighted)** | 0.5167 |

## Latency
- **Avg Duration**: 12633.42 ms
- **Total Duration**: 6316.71 s

## Sample Results (First 10)
|   ddc_code | predicted_codes                     | title                                                          |
|-----------:|:------------------------------------|:---------------------------------------------------------------|
|        332 | ['330', '336']                      | Dollar Drops Further as Central Banks Reassess Reserves        |
|        500 | ['502', '509', '506', '507', '505'] | Sony Shows Smaller PlayStation 2 (AP)                          |
|        500 | ['506', '505', '507', '509', '502'] | Phishing on the increase, group says (InfoWorld)               |
|        796 | ['796', '790', '791', '792', '794'] | Woods struggling to cope with body changes: Singh              |
|        500 | ['500', '501', '502', '506', '507'] | Intel shelves plans for Wi-Fi access point                     |
|        500 | ['502', '506', '509', '507', '505'] | Convicted spammer gets nine years in slammer                   |
|        796 | ['796', '790', '791', '792', '793'] | Bills' Williams Sustains Neck Injury (AP)                      |
|        330 | ['330', '336']                      | China vows currency shift but mum on date                      |
|        796 | ['796', '791', '790', '792', '793'] | We owe Athens an apology                                       |
|        355 | ['351', '358', '355']               | Abductor kills self in Moscow region hostage freeing operation |