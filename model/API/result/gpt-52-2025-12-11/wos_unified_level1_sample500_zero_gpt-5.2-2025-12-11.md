# Evaluation Report
**Level**: 1
**Dataset**: wos_unified.csv
**Shot Type**: zero
**Sample Size**: 500
**Model**: gpt-5.2-2025-12-11

## Accuracy Metrics
| Metric | Value | Count |
|--------|-------|-------|
| **Top-1** | 40.20% | 201/500 |
| **Top-3** | 86.20% | 431/500 |
| **Top-5** | 98.00% | 490/500 |

## Weighted Metrics (Top-1)
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.4020 |
| **Precision (Weighted)** | 0.8598 |
| **Recall (Weighted)** | 0.4020 |
| **F1 Score (Weighted)** | 0.4340 |

## Latency
- **Avg Duration**: 10346.91 ms
- **Total Duration**: 5173.45 s

## Sample Results (First 10)
|   ddc_code | predicted_codes                     | title                                                                                                                             |
|-----------:|:------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------|
|        000 | ['500', '000', '600', '900', 'DDC'] | scientific computation; parallel computation; geodynamics; mid-ocean ridge; PETSc                                                 |
|        600 | ['500', '300', '100', '600', '000'] | sleep; operant extinction; recent memory; remote memory; generalization; context                                                  |
|        500 | ['500', '600', '000', '300', '100'] | Fibroblast; Aloe vera (A.v); bFGF; TGF beta 1                                                                                     |
|        600 | ['500', '600', '300', '100', '000'] | inflammation; sports injury; omega-3 fatty acids; resolvins                                                                       |
|        600 | ['500', '300', '100', '400', '600'] | Autism spectrum disorder; Pervasive developmental disorder; Developmental and speech delay                                        |
|        600 | ['500', '300', '600', '900', '100'] | Prevalence; Pain; HIV/AIDS                                                                                                        |
|        600 | ['500', '600', '300', '000', '900'] | Lung cancer; PD-L1; PD-1; Pembrolizumab; EGFR; ALK                                                                                |
|        600 | ['500', '300', '600', '000', '100'] | Emergency contraception; Ulipristal; Copper IUD; Levonorgestrel                                                                   |
|        600 | ['500', '300', '100', '600', '900'] | Chronic obstructive pulmonary disease; Idiopathic pulmonary fibrosis; Lung cancer; Amyotrophic lateral sclerosis; Palliative care |
|        600 | ['000', '600', '300', '500', 'DDC'] | Multi-objective analysis; Artificial bee colony; Differential evolution; Time-cost-quality tradeoff; Construction management      |