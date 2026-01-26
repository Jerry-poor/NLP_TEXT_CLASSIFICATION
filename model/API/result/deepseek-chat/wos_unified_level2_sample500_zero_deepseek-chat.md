# Evaluation Report
**Level**: 2
**Dataset**: wos_unified.csv
**Shot Type**: zero
**Sample Size**: 500
**Model**: deepseek-chat

## Accuracy Metrics
| Metric | Value | Count |
|--------|-------|-------|
| **Top-1** | 71.20% | 356/500 |
| **Top-3** | 94.00% | 470/500 |
| **Top-5** | 99.00% | 495/500 |

## Weighted Metrics (Top-1)
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.7120 |
| **Precision (Weighted)** | 0.9704 |
| **Recall (Weighted)** | 0.7120 |
| **F1 Score (Weighted)** | 0.7801 |

## Latency
- **Avg Duration**: 12397.06 ms
- **Total Duration**: 6198.53 s

## Sample Results (First 10)
|   ddc_code | predicted_codes                     | title                                                                                                                                                       |
|-----------:|:------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------|
|        540 | ['570', '580', '540', '500', '530'] | Fibroblast; Aloe vera (A.v); bFGF; TGF beta 1                                                                                                               |
|        610 | ['620', '600', '680', '610']        | Affinity tag; Fc; Fc-fusion; recombinant protein; stability                                                                                                 |
|        610 | ['610', '600', '650', '640', '630'] | Prevalence; Pain; HIV/AIDS                                                                                                                                  |
|        540 | ['570', '500', '540', '530', '510'] | Allogenic lymphocyte immunotherapy; effectiveness; meta-analysis; timing of treatment; unexplained recurrent spontaneous abortion                           |
|        610 | ['610', '600', '620', '630', '640'] | sleep; operant extinction; recent memory; remote memory; generalization; context                                                                            |
|        610 | ['610', '600', '620', '670']        | Chronic rhinosinusitis with nasal polyps; aspirin-exacerbated respiratory disease; Staphylococcus aureus enterotoxin; superantigen; superantibody; basophil |
|        620 | ['620', '600', '680', '670']        | analog integrated circuits; interface of sensors; structured array; radiation hardness; operational amplifier; amplifier with current negative feedback     |
|        610 | ['610', '600', '650', '680', '620'] | Emergency contraception; Ulipristal; Copper IUD; Levonorgestrel                                                                                             |
|        540 | ['540', '570', '500', '530', '510'] | acrylamide; glycidamide; human metabolism; toxicokinetics; human exposure; daily intake; mercapturic acids; hemoglobin adducts                              |
|        620 | ['690', '620', '650', '600', '670'] | Multi-objective analysis; Artificial bee colony; Differential evolution; Time-cost-quality tradeoff; Construction management                                |