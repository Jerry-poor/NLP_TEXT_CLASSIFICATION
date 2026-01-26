# Evaluation Report
**Level**: 2
**Dataset**: wos_unified.csv
**Shot Type**: zero
**Sample Size**: 500
**Model**: gpt-5.2-2025-12-11

## Accuracy Metrics
| Metric | Value | Count |
|--------|-------|-------|
| **Top-1** | 78.60% | 393/500 |
| **Top-3** | 96.00% | 480/500 |
| **Top-5** | 99.20% | 496/500 |

## Weighted Metrics (Top-1)
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.7860 |
| **Precision (Weighted)** | 0.9830 |
| **Recall (Weighted)** | 0.7860 |
| **F1 Score (Weighted)** | 0.8378 |

## Latency
- **Avg Duration**: 10311.05 ms
- **Total Duration**: 5155.53 s

## Sample Results (First 10)
|   ddc_code | predicted_codes                     | title                                                                                                                                                                            |
|-----------:|:------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|        610 | ['610', '600', '620', '680']        | Matrix Metaloproteinase-1; Zoanthamine alkaloid class; Docking; Molecular dynamics simulation; MM-PB/GBSA                                                                        |
|        000 | ['000', '020', '070', '050', '060'] | Cloud computing; Building information model; Big data analysis; Web3D; Bigtable; MapReduce                                                                                       |
|        620 | ['620', '600', '670', '690', '650'] | nonlinear systems; flatness; exact linearization; tracking control; disturbance rejection                                                                                        |
|        620 | ['620', '600', '670', '650']        | Distributed wind energy; Wind turbine gearbox; Variable speed operation; Aerodynamic efficiency                                                                                  |
|        620 | ['600', '650', '620', '630']        | Fracking; energy; government subsidies; property rights violations                                                                                                               |
|        000 | ['000', '020', '010', '050', '070'] | Topic modeling; LDA; LSI; Survey                                                                                                                                                 |
|        000 | ['000', '020', '070', '050', '060'] | Public-transport network; General transit feed specification; Network topology; Node-degree distribution; Average (shortest) path length; Average network clustering coefficient |
|        540 | ['540', '570', '500', '530', '510'] | acrylamide; glycidamide; human metabolism; toxicokinetics; human exposure; daily intake; mercapturic acids; hemoglobin adducts                                                   |
|        610 | ['610', '650', '600', '640', '620'] | Prevalence; Pain; HIV/AIDS                                                                                                                                                       |
|        610 | ['610', '600', '650', '620', '630'] | sleep; operant extinction; recent memory; remote memory; generalization; context                                                                                                 |