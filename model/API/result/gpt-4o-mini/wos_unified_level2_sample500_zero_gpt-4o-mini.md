# Evaluation Report
**Level**: 2
**Dataset**: wos_unified.csv
**Shot Type**: zero
**Sample Size**: 500
**Model**: gpt-4o-mini

## Accuracy Metrics
| Metric | Value | Count |
|--------|-------|-------|
| **Top-1** | 68.40% | 342/500 |
| **Top-3** | 82.20% | 411/500 |
| **Top-5** | 89.20% | 446/500 |

## Weighted Metrics (Top-1)
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.6840 |
| **Precision (Weighted)** | 0.9871 |
| **Recall (Weighted)** | 0.6840 |
| **F1 Score (Weighted)** | 0.7766 |

## Latency
- **Avg Duration**: 18157.86 ms
- **Total Duration**: 9078.93 s

## Sample Results (First 10)
|   ddc_code | predicted_codes                     | title                                                                                                                                                             |
|-----------:|:------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|        620 | ['600', '620', '650', '670', '640'] | Human-computer interaction; Ambient-intelligence; Fuzzy-logic; User interfaces                                                                                    |
|        610 | ['610', '620', '600', '650', '670'] | complications (lung clinical); fungal (infection); quality of life (quality of life    ethics    economics); solid tumor (malignancy and long-term complications) |
|        540 | ['540', '570', '500', '550', '590'] | acrylamide; glycidamide; human metabolism; toxicokinetics; human exposure; daily intake; mercapturic acids; hemoglobin adducts                                    |
|        540 | ['570', '580', '590', '500', '540'] | Fibroblast; Aloe vera (A.v); bFGF; TGF beta 1                                                                                                                     |
|        620 | ['620', '600', '670', '610']        | Nanofiber membrane; Electrospinning; Boron removal; Surface grafting; Adsorption                                                                                  |
|        610 | ['610', '600', '620', '670', '650'] | Affinity tag; Fc; Fc-fusion; recombinant protein; stability                                                                                                       |
|        610 | ['610', '600', '620', '650', '640'] | Chronic rhinosinusitis with nasal polyps; aspirin-exacerbated respiratory disease; Staphylococcus aureus enterotoxin; superantigen; superantibody; basophil       |
|        610 | ['610', '600', '650', '640', '630'] | sleep; operant extinction; recent memory; remote memory; generalization; context                                                                                  |
|        610 | ['610', '600', '650', '620', '670'] | Viral load; host genetic factors; viral diagnostic; syndromic diagnosis; point of care; HIV; HCV; HPV; RSV; IL28B; ITPA                                           |
|        540 | ['570', '580', '500', '540', '590'] | genome editing; targeted mutagenesis; engineered nucleases; double-strand break repair; homologous recombination; technical advance                               |