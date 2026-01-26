# Evaluation Report
**Level**: 2
**Dataset**: lib_unified.csv
**Shot Type**: zero
**Sample Size**: 500
**Model**: deepseek-chat

## Accuracy Metrics
| Metric | Value | Count |
|--------|-------|-------|
| **Top-1** | 61.80% | 309/500 |
| **Top-3** | 83.60% | 418/500 |
| **Top-5** | 90.60% | 453/500 |

## Weighted Metrics (Top-1)
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.6180 |
| **Precision (Weighted)** | 0.7791 |
| **Recall (Weighted)** | 0.6180 |
| **F1 Score (Weighted)** | 0.6635 |

## Latency
- **Avg Duration**: 13186.15 ms
- **Total Duration**: 6593.07 s

## Sample Results (First 10)
|   ddc_code | predicted_codes                     | title                                                           |
|-----------:|:------------------------------------|:----------------------------------------------------------------|
|        350 | ['320', '350', '300', '340', '380'] | Rwanda denies army in Congo                                     |
|        320 | ['380', '300', '330', '390', '360'] | Benitez confirms Morientes interest (AFP)                       |
|        300 | ['380', '370', '300', '360', '390'] | The future of writing                                           |
|        330 | ['380', '330', '300', '350', '340'] | Halliburton may shed KBR unit                                   |
|        500 | ['500', '510', '520', '530', '540'] | New bug in open source database MySQL                           |
|        570 | ['570', '500', '540', '530', '510'] | Anxiety                                                         |
|        620 | ['690', '650', '600', '620', '670'] | Construction Management                                         |
|        330 | ['380', '330', '310', '300', '340'] | Investors in StarHub sell shares                                |
|        610 | ['600', '620', '670', '650']        | network security                                                |
|        500 | ['500', '510', '520', '530', '540'] | Media 100 HD video editing system released for Mac (MacCentral) |