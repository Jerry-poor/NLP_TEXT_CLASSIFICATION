# Evaluation Report
**Level**: 2
**Dataset**: ag_news_unified.csv
**Shot Type**: zero
**Sample Size**: 500
**Model**: gpt-5.2-2025-12-11

## Accuracy Metrics
| Metric | Value | Count |
|--------|-------|-------|
| **Top-1** | 61.40% | 307/500 |
| **Top-3** | 88.40% | 442/500 |
| **Top-5** | 95.20% | 476/500 |

## Weighted Metrics (Top-1)
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.6140 |
| **Precision (Weighted)** | 0.8228 |
| **Recall (Weighted)** | 0.6140 |
| **F1 Score (Weighted)** | 0.6514 |

## Latency
- **Avg Duration**: 10425.52 ms
- **Total Duration**: 5212.76 s

## Sample Results (First 10)
|   ddc_code | predicted_codes                     | title                                                                               |
|-----------:|:------------------------------------|:------------------------------------------------------------------------------------|
|        790 | ['790', '780', '770', '700', '710'] | Davis Cup: Australia takes 2-0 lead in World Group playoff                          |
|        320 | ['320', '340', '350', '300', '330'] | Japanese Prime Minister Inspects Four Northern Islands under &lt;b&gt;...&lt;/b&gt; |
|        500 | ['510', '500', '530', '540', '550'] | EDS Is Charter Member of Siebel BPO Alliance (NewsFactor)                           |
|        790 | ['790', '780', '770', '700']        | Fan v Fan: Manchester City-Tottenham Hotspur                                        |
|        650 | ['650', '600', '670', '620', '680'] | Oracle Quarterly Net Income Rises 16 Pct                                            |
|        320 | ['390', '380', '300', '330', '370'] | Paris Tourists Search for Key to 'Da Vinci Code' (Reuters)                          |
|        790 | ['790', '780', '770', '700']        | Premiership: Charlton snatch win                                                    |
|        790 | ['790', '780', '770', '700']        | Giants Give Up Right to Void Bonds' Deal (AP)                                       |
|        790 | ['790', '780', '770', '700']        | Tributes flood in for Nicholson                                                     |
|        330 | ['330', '380', '310', '340', '300'] | Profit Plunges at International Game Tech                                           |