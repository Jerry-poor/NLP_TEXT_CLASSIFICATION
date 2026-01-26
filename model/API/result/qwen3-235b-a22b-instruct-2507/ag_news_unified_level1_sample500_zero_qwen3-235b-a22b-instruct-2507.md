# Evaluation Report
**Level**: 1
**Dataset**: ag_news_unified.csv
**Shot Type**: zero
**Sample Size**: 500
**Model**: qwen3-235b-a22b-instruct-2507

## Accuracy Metrics
| Metric | Value | Count |
|--------|-------|-------|
| **Top-1** | 61.20% | 306/500 |
| **Top-3** | 87.80% | 439/500 |
| **Top-5** | 92.60% | 463/500 |

## Weighted Metrics (Top-1)
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.6120 |
| **Precision (Weighted)** | 0.8184 |
| **Recall (Weighted)** | 0.6120 |
| **F1 Score (Weighted)** | 0.6710 |

## Latency
- **Avg Duration**: 11567.84 ms
- **Total Duration**: 5783.92 s

## Sample Results (First 10)
|   ddc_code | predicted_codes                     | title                                                      |
|-----------:|:------------------------------------|:-----------------------------------------------------------|
|        700 | ['900', '300', '700', '800', '400'] | Tributes flood in for Nicholson                            |
|        700 | ['700', '900', '300', '400', 'DDC'] | Davis Cup: Australia takes 2-0 lead in World Group playoff |
|        300 | ['300', '900', '200', '100', '400'] | Palestinians kill three Israeli soldiers                   |
|        300 | ['600', '300', '500', '000', 'DDC'] | Crunch Time for Biotech Companies                          |
|        700 | ['900', '300', '400', '700', '000'] | United Apology over Website Abuse                          |
|        700 | ['700', '900', '300', '400', 'DDC'] | One assist, goal for hometown star                         |
|        300 | ['900', '800', '700', '300', '200'] | Paris Tourists Search for Key to 'Da Vinci Code' (Reuters) |
|        300 | ['300', '600', '000', 'DDC']        | Fannie Mae mess worries investors                          |
|        600 | ['300', '000', '600', '500', '900'] | Oracle Quarterly Net Income Rises 16 Pct                   |
|        700 | ['700', '900', '300', '400', 'DDC'] | GAME DAY RECAP Thursday, September 09                      |