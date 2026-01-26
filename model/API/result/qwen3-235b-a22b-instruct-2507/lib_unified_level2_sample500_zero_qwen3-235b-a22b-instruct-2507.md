# Evaluation Report
**Level**: 2
**Dataset**: lib_unified.csv
**Shot Type**: zero
**Sample Size**: 500
**Model**: qwen3-235b-a22b-instruct-2507

## Accuracy Metrics
| Metric | Value | Count |
|--------|-------|-------|
| **Top-1** | 64.80% | 324/500 |
| **Top-3** | 84.20% | 421/500 |
| **Top-5** | 90.60% | 453/500 |

## Weighted Metrics (Top-1)
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.6480 |
| **Precision (Weighted)** | 0.7443 |
| **Recall (Weighted)** | 0.6480 |
| **F1 Score (Weighted)** | 0.6736 |

## Latency
- **Avg Duration**: 10742.44 ms
- **Total Duration**: 5371.22 s

## Sample Results (First 10)
|   ddc_code | predicted_codes                     | title                                                           |
|-----------:|:------------------------------------|:----------------------------------------------------------------|
|        790 | ['790', '700', '710', '720', '780'] | Minnesota Timberwolves Center Arrested (AP)                     |
|        790 | ['790', '710', '720', '700', '770'] | American League Game Summary - Cleveland At Kansas City         |
|        500 | ['500', '510', '530', '540', '550'] | Briefly: Dell updates low-end server line                       |
|        790 | ['790', '710', '720', '700', '780'] | Sox agree to terms with Hermanson                               |
|        500 | ['500', '510', '530', '550', '570'] | Media 100 HD video editing system released for Mac (MacCentral) |
|        610 | ['600', '620', '650', '680']        | network security                                                |
|        320 | ['370', '360', '320', '350', '340'] | Children Return to Classes in Russia                            |
|        330 | ['330', '380', '320', '350', '360'] | Halliburton may shed KBR unit                                   |
|        790 | ['790', '710', '720', '700', '780'] | Lions prepare for Vick, Falcons                                 |
|        330 | ['330', '380', '310', '300', '360'] | Weyerhaeuser 3rd-Quarter Earnings Rise                          |