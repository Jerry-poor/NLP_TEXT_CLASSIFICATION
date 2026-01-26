# Evaluation Report
**Level**: 1
**Dataset**: wos_unified.csv
**Shot Type**: zero
**Sample Size**: 500
**Model**: qwen3-235b-a22b-instruct-2507

## Accuracy Metrics
| Metric | Value | Count |
|--------|-------|-------|
| **Top-1** | 44.80% | 224/500 |
| **Top-3** | 88.00% | 440/500 |
| **Top-5** | 94.20% | 471/500 |

## Weighted Metrics (Top-1)
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.4480 |
| **Precision (Weighted)** | 0.8259 |
| **Recall (Weighted)** | 0.4480 |
| **F1 Score (Weighted)** | 0.4908 |

## Latency
- **Avg Duration**: 11797.42 ms
- **Total Duration**: 5898.71 s

## Sample Results (First 10)
|   ddc_code | predicted_codes                     | title                                                                            |
|-----------:|:------------------------------------|:---------------------------------------------------------------------------------|
|        600 | ['600', '500', '000', '300', '900'] | Nanofiber membrane; Electrospinning; Boron removal; Surface grafting; Adsorption |
|        600 | ['300', '500', '100', '900', '600'] | sleep; operant extinction; recent memory; remote memory; generalization; context |
|        500 | ['500', '600', '300', '000', 'DDC'] | XET; litchi fruit; cracking; growth; NAA                                         |
|        600 | ['500', '600', '300', '000']        | rheumatoid arthritis; in vivo hypoxia imaging; PET; HIF; ROS                     |
|        600 | ['500', '600', '300', '000']        | classification; lymphoma; World Health Organization                              |
|        600 | ['600', '500', '000', '300', '700'] | Affinity tag; Fc; Fc-fusion; recombinant protein; stability                      |
|        500 | ['500', '600', '300', '900']        | Fibroblast; Aloe vera (A.v); bFGF; TGF beta 1                                    |
|        600 | ['300', '500', '100', '600', 'DDC'] | perceived stress; stress; headache; adolescence; Ecological Momentary Assessment |
|        600 | ['000', '600', '500', '300', '700'] | Human-computer interaction; Ambient-intelligence; Fuzzy-logic; User interfaces   |
|        600 | ['600', '500', '000', '700']        | current-mode; frequency filter KHN equivalent; ECCII; analog signal processing   |