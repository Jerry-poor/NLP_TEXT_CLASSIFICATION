# Evaluation Report
**Level**: 2
**Dataset**: ag_news_unified.csv
**Shot Type**: zero
**Sample Size**: 500
**Model**: deepseek-chat

## Accuracy Metrics
| Metric | Value | Count |
|--------|-------|-------|
| **Top-1** | 55.20% | 276/500 |
| **Top-3** | 85.80% | 429/500 |
| **Top-5** | 93.00% | 465/500 |

## Weighted Metrics (Top-1)
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.5520 |
| **Precision (Weighted)** | 0.8565 |
| **Recall (Weighted)** | 0.5520 |
| **F1 Score (Weighted)** | 0.6215 |

## Latency
- **Avg Duration**: 12460.01 ms
- **Total Duration**: 6230.00 s

## Sample Results (First 10)
|   ddc_code | predicted_codes                     | title                                                      |
|-----------:|:------------------------------------|:-----------------------------------------------------------|
|        330 | ['320', '300', '340', '350', '330'] | Chavez rejects CD as opposition                            |
|        330 | ['330', '360', '380', '310', '300'] | Mortgage approvals drop sharply                            |
|        790 | ['790', '700', '780', '770']        | Baseball-Red Sox on Brink of World Series Victory          |
|        330 | ['380', '330', '300', '310', '350'] | Clouds darken PeopleSoft conference                        |
|        320 | ['390', '300', '370', '380', '360'] | Paris Tourists Search for Key to 'Da Vinci Code' (Reuters) |
|        000 | ['070', '050', '030', '060', '010'] | Salvaging Genesis                                          |
|        500 | ['500', '510', '520', '530', '540'] | Net firms: Don't tax VoIP                                  |
|        320 | ['320', '300', '340', '350', '360'] | Ukrainian opposition makes gains                           |
|        330 | ['340', '330', '380', '300', '350'] | Computer Associates exec pleads not guilty                 |
|        790 | ['790', '700', '770', '710']        | United Apology over Website Abuse                          |