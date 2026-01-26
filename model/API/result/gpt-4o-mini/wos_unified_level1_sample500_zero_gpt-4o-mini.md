# Evaluation Report
**Level**: 1
**Dataset**: wos_unified.csv
**Shot Type**: zero
**Sample Size**: 500
**Model**: gpt-4o-mini

## Accuracy Metrics
| Metric | Value | Count |
|--------|-------|-------|
| **Top-1** | 37.80% | 189/500 |
| **Top-3** | 82.60% | 413/500 |
| **Top-5** | 93.60% | 468/500 |

## Weighted Metrics (Top-1)
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.3780 |
| **Precision (Weighted)** | 0.7735 |
| **Recall (Weighted)** | 0.3780 |
| **F1 Score (Weighted)** | 0.4138 |

## Latency
- **Avg Duration**: 15596.21 ms
- **Total Duration**: 7798.11 s

## Sample Results (First 10)
|   ddc_code | predicted_codes                     | title                                                                                                                             |
|-----------:|:------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------|
|        500 | ['500', '600', '300', '100', '900'] | Fibroblast; Aloe vera (A.v); bFGF; TGF beta 1                                                                                     |
|        600 | ['500', '600', '300', '100', '900'] | Lung cancer; PD-L1; PD-1; Pembrolizumab; EGFR; ALK                                                                                |
|        600 | ['500', '600', '300', '100', '000'] | parkinsonism; MPTP; caffeine; DPCPX; KW-6002                                                                                      |
|        600 | ['300', '500', '100', '600', '900'] | Chronic obstructive pulmonary disease; Idiopathic pulmonary fibrosis; Lung cancer; Amyotrophic lateral sclerosis; Palliative care |
|        600 | ['600', '300', '500', '900', '000'] | Fracking; energy; government subsidies; property rights violations                                                                |
|        600 | ['600', '000', '500', '300', '900'] | current-mode; frequency filter KHN equivalent; ECCII; analog signal processing                                                    |
|        000 | ['000', '600', '500', '300', '900'] | CUDA; FFT; GPGPU; GPU computing; many-core architecture; micromagnetics; OpenCL; parallel computing                               |
|        000 | ['000', '300', '600', '100', '900'] | Topic modeling; LDA; LSI; Survey                                                                                                  |
|        600 | ['500', '300', '600', '100', '000'] | Bifidobacterium; brain gut axis; GABA; microbiome; neuromodulation                                                                |
|        500 | ['500', '600', '300', '100', '900'] | Cancer; carbohydrate recognition domain; galectin-9; immune-regulation; linker peptide                                            |