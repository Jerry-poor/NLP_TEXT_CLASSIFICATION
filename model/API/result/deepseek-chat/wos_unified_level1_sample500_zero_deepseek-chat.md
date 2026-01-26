# Evaluation Report
**Level**: 1
**Dataset**: wos_unified.csv
**Shot Type**: zero
**Sample Size**: 500
**Model**: deepseek-chat

## Accuracy Metrics
| Metric | Value | Count |
|--------|-------|-------|
| **Top-1** | 45.40% | 227/500 |
| **Top-3** | 93.20% | 466/500 |
| **Top-5** | 99.20% | 496/500 |

## Weighted Metrics (Top-1)
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.4540 |
| **Precision (Weighted)** | 0.7678 |
| **Recall (Weighted)** | 0.4540 |
| **F1 Score (Weighted)** | 0.5037 |

## Latency
- **Avg Duration**: 12066.23 ms
- **Total Duration**: 6033.11 s

## Sample Results (First 10)
|   ddc_code | predicted_codes                     | title                                                                                                                                                                            |
|-----------:|:------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|        600 | ['100', '500', '300', '600', '400'] | Autism spectrum disorder; Pervasive developmental disorder; Developmental and speech delay                                                                                       |
|        000 | ['000', '600', '500', '300', '100'] | Attack graph; alert correlation; network hardening; security metric                                                                                                              |
|        600 | ['600', '500', '000', '300', '700'] | Distributed wind energy; Wind turbine gearbox; Variable speed operation; Aerodynamic efficiency                                                                                  |
|        000 | ['000', '600', '500', '300', '900'] | Public-transport network; General transit feed specification; Network topology; Node-degree distribution; Average (shortest) path length; Average network clustering coefficient |
|        600 | ['600', '500', '000', '100', '300'] | nonlinear systems; flatness; exact linearization; tracking control; disturbance rejection                                                                                        |
|        600 | ['600', '500', '000', '700', '900'] | current-mode; frequency filter KHN equivalent; ECCII; analog signal processing                                                                                                   |
|        500 | ['500', '600', '000', '100', '200'] | XET; litchi fruit; cracking; growth; NAA                                                                                                                                         |
|        600 | ['300', '500', '600', '100', '900'] | Prevalence; Pain; HIV/AIDS                                                                                                                                                       |
|        500 | ['500', '600', '300', '100', '000'] | acrylamide; glycidamide; human metabolism; toxicokinetics; human exposure; daily intake; mercapturic acids; hemoglobin adducts                                                   |
|        600 | ['600', '500', '300', '900', '000'] | Phyllite; Shear creep; Softening effect; Creep parameters                                                                                                                        |