# Evaluation Report
**Level**: 1
**Dataset**: ag_news_unified.csv
**Shot Type**: zero
**Sample Size**: 500
**Model**: deepseek-chat

## Accuracy Metrics
| Metric | Value | Count |
|--------|-------|-------|
| **Top-1** | 64.00% | 320/500 |
| **Top-3** | 90.40% | 452/500 |
| **Top-5** | 96.80% | 484/500 |

## Weighted Metrics (Top-1)
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.6400 |
| **Precision (Weighted)** | 0.8215 |
| **Recall (Weighted)** | 0.6400 |
| **F1 Score (Weighted)** | 0.6893 |

## Latency
- **Avg Duration**: 14058.51 ms
- **Total Duration**: 7029.25 s

## Sample Results (First 10)
|   ddc_code | predicted_codes                     | title                                                        |
|-----------:|:------------------------------------|:-------------------------------------------------------------|
|        300 | ['900', '300', '200', '100', '400'] | Palestinians kill three Israeli soldiers                     |
|        700 | ['700', '300', '900', '400', '500'] | Premiership: Charlton snatch win                             |
|        300 | ['300', '600', '500', '000', '900'] | Sales boost for House of Fraser                              |
|        500 | ['600', '300', '000', '500', '900'] | EDS Is Charter Member of Siebel BPO Alliance (NewsFactor)    |
|        300 | ['300', '600', '500', '000', '900'] | Gateway Updates 4Q, Year Guidance                            |
|        300 | ['300', '900', '200', '700', '400'] | Thais  #39;bomb #39; south with paper birds on Muslim south  |
|        000 | ['300', '600', '000', '500', '900'] | Some People Not Eligible to Get in on Google IPO             |
|        700 | ['700', '300', '900', '400', '500'] | Fan v Fan: Manchester City-Tottenham Hotspur                 |
|        300 | ['300', '900', '100', '400', '200'] | Prince Charles chastised for  quot;old fashioned quot; views |
|        300 | ['300', '600', '900', '100', '400'] | Computer Associates exec pleads not guilty                   |