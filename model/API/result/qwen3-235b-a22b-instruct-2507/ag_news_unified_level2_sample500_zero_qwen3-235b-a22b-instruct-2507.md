# Evaluation Report
**Level**: 2
**Dataset**: ag_news_unified.csv
**Shot Type**: zero
**Sample Size**: 500
**Model**: qwen3-235b-a22b-instruct-2507

## Accuracy Metrics
| Metric | Value | Count |
|--------|-------|-------|
| **Top-1** | 61.40% | 307/500 |
| **Top-3** | 86.00% | 430/500 |
| **Top-5** | 93.60% | 468/500 |

## Weighted Metrics (Top-1)
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.6140 |
| **Precision (Weighted)** | 0.7924 |
| **Recall (Weighted)** | 0.6140 |
| **F1 Score (Weighted)** | 0.6465 |

## Latency
- **Avg Duration**: 11207.74 ms
- **Total Duration**: 5603.87 s

## Sample Results (First 10)
|   ddc_code | predicted_codes                     | title                                        |
|-----------:|:------------------------------------|:---------------------------------------------|
|        500 | ['500', '510', '530', '550', '570'] | Net firms: Don't tax VoIP                    |
|        330 | ['380', '330', '300', '350', '360'] | General Mills goes whole grains              |
|        790 | ['790', '700', '780', '770']        | Premiership: Charlton snatch win             |
|        500 | ['590', '570', '580', '560', '550'] | Dependent species risk extinction            |
|        000 | ['070', '090', '030', '010', '080'] | Salvaging Genesis                            |
|        790 | ['790', '700', '710', '720']        | GAME DAY RECAP Thursday, September 09        |
|        790 | ['790', '700', '710', '720', '730'] | United Apology over Website Abuse            |
|        790 | ['790', '710', '720', '700', '780'] | NCAA Wrong To Close Book On Williams         |
|        330 | ['330', '300', '380', '310', '360'] | Mortgage approvals drop sharply              |
|        790 | ['790', '700', '770', '780']        | Fan v Fan: Manchester City-Tottenham Hotspur |