# Evaluation Report
**Level**: 1
**Dataset**: ag_news_unified.csv
**Shot Type**: zero
**Sample Size**: 500
**Model**: gpt-4o-mini

## Accuracy Metrics
| Metric | Value | Count |
|--------|-------|-------|
| **Top-1** | 51.40% | 257/500 |
| **Top-3** | 76.80% | 384/500 |
| **Top-5** | 84.00% | 420/500 |

## Weighted Metrics (Top-1)
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.5140 |
| **Precision (Weighted)** | 0.7008 |
| **Recall (Weighted)** | 0.5140 |
| **F1 Score (Weighted)** | 0.5129 |

## Latency
- **Avg Duration**: 16035.50 ms
- **Total Duration**: 8017.75 s

## Sample Results (First 10)
|   ddc_code | predicted_codes                     | title                                                 |
|-----------:|:------------------------------------|:------------------------------------------------------|
|        300 | ['300', '600', '500', '900', '000'] | Campbell 9 Pct. Profit  #39;Hmmm Hmmm Good #39;       |
|        000 | ['600', '000', '500', '300', '900'] | Linux groups patch image flaw                         |
|        300 | ['600', '500', '300', '900', '000'] | General Mills goes whole grains                       |
|        000 | ['500', '600', '900', '000', '300'] | Salvaging Genesis                                     |
|        700 | ['300', '900', '700', '000', '800'] | Giants Give Up Right to Void Bonds' Deal (AP)         |
|        300 | ['300', '100', '900', '000', '800'] | Forgoing stiff upper lip, Charles jousts with critics |
|        300 | ['300', '900', '600', '000', '100'] | Mortgage approvals drop sharply                       |
|        300 | ['300', '900', '100', '000', '600'] | Ukrainian opposition makes gains                      |
|        300 | ['300', '900', '100', '000', '200'] | Chavez rejects CD as opposition                       |
|        700 | ['900', '300', '700', '500', '600'] | Strongwoman hoists 100th gold for Chinese delegation  |