# Evaluation Report
**Level**: 2
**Dataset**: wos_unified.csv
**Shot Type**: zero
**Sample Size**: 500
**Model**: qwen3-235b-a22b-instruct-2507

## Accuracy Metrics
| Metric | Value | Count |
|--------|-------|-------|
| **Top-1** | 76.60% | 383/500 |
| **Top-3** | 92.00% | 460/500 |
| **Top-5** | 98.00% | 490/500 |

## Weighted Metrics (Top-1)
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.7660 |
| **Precision (Weighted)** | 0.9784 |
| **Recall (Weighted)** | 0.7660 |
| **F1 Score (Weighted)** | 0.8224 |

## Latency
- **Avg Duration**: 18482.78 ms
- **Total Duration**: 9241.39 s

## Sample Results (First 10)
|   ddc_code | predicted_codes                     | title                                                                                                                                                             |
|-----------:|:------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|        610 | ['610', '620', '650', '600', '630'] | ANKYLOSING SPONDYLITIS; SPONDYLOARTHRITIS; OUTCOMES; TUMOR NECROSIS FACTOR INHIBITORS                                                                             |
|        610 | ['610', '620', '600', '680', '650'] | Affinity tag; Fc; Fc-fusion; recombinant protein; stability                                                                                                       |
|        610 | ['610', '620', '600', '650', '630'] | complications (lung clinical); fungal (infection); quality of life (quality of life    ethics    economics); solid tumor (malignancy and long-term complications) |
|        620 | ['620', '600', '670', '610']        | Distributed wind energy; Wind turbine gearbox; Variable speed operation; Aerodynamic efficiency                                                                   |
|        620 | ['620', '600', '670', '680']        | nonlinear systems; flatness; exact linearization; tracking control; disturbance rejection                                                                         |
|        000 | ['000', '020', '010', '050', '070'] | Attack graph; alert correlation; network hardening; security metric                                                                                               |
|        150 | ['150', '170', '140', '100', '120'] | CEO deaths; CEO effects; event study; managerial discretion; senior executives                                                                                    |
|        000 | ['000', '020', '070', '060', '050'] | scientific computation; parallel computation; geodynamics; mid-ocean ridge; PETSc                                                                                 |
|        620 | ['620', '600', '670', '690']        | Phyllite; Shear creep; Softening effect; Creep parameters                                                                                                         |
|        000 | ['000', '020', '050', '070', '060'] | Cloud computing; Building information model; Big data analysis; Web3D; Bigtable; MapReduce                                                                        |