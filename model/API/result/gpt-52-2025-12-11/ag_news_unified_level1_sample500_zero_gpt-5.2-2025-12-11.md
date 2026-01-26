# Evaluation Report
**Level**: 1
**Dataset**: ag_news_unified.csv
**Shot Type**: zero
**Sample Size**: 500
**Model**: gpt-5.2-2025-12-11

## Accuracy Metrics
| Metric | Value | Count |
|--------|-------|-------|
| **Top-1** | 63.80% | 319/500 |
| **Top-3** | 89.00% | 445/500 |
| **Top-5** | 94.00% | 470/500 |

## Weighted Metrics (Top-1)
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.6380 |
| **Precision (Weighted)** | 0.7708 |
| **Recall (Weighted)** | 0.6380 |
| **F1 Score (Weighted)** | 0.6736 |

## Latency
- **Avg Duration**: 10636.42 ms
- **Total Duration**: 5318.21 s

## Sample Results (First 10)
|   ddc_code | predicted_codes                     | title                                                        |
|-----------:|:------------------------------------|:-------------------------------------------------------------|
|        300 | ['300', '900', '600', '000', '500'] | Mortgage approvals drop sharply                              |
|        300 | ['900', '300', '800', '100', '200'] | Forgoing stiff upper lip, Charles jousts with critics        |
|        300 | ['000', '300', '600', '900', '800'] | Clouds darken PeopleSoft conference                          |
|        300 | ['900', '300', '200', '600', '700'] | Thais  #39;bomb #39; south with paper birds on Muslim south  |
|        300 | ['900', '300', '800', '700', '400'] | Selling Houston Warts and All, Especially Warts              |
|        300 | ['300', '600', '000', '900', '700'] | Profit Plunges at International Game Tech                    |
|        500 | ['000', '600', '300', '900', '400'] | EDS Is Charter Member of Siebel BPO Alliance (NewsFactor)    |
|        500 | ['500', '300', '600', '900', '100'] | Dependent species risk extinction                            |
|        700 | ['300', '900', '000', '800', '700'] | United Apology over Website Abuse                            |
|        300 | ['300', '900', '400', '100', '800'] | Prince Charles chastised for  quot;old fashioned quot; views |