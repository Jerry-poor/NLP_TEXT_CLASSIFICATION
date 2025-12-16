# NER Experiment Report

## Summary
ds_v5固定随机种子，开启了 cuDNN 的确定性，几乎是可以复现的。重新训练了 **1.5B** 模型。

**Token Evaluation Note**: 所有指标 token 只计算非 O token。

---

## 1. DeepSeek-R1 Models

### DeepSeek-R1 1.5B (v5/Reproducible)
| Metric Type | Total | Correct/TP | Incorrect/FP | FN | Precision/Acc | Recall | F1-score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sentence Level** | 1000 | 762 | - | - | 0.7620 | - | - |
| **Entity Level** | 2328 | 2027 | 299 | 301 | 0.8715 (Acc) | 0.8707 | **0.8711** |
| **Token Level** | 3307 | 2800 | - | - | 0.8475 | 0.8467 | 0.8471 |

### DeepSeek-R1 7B
| Metric Type | Total | Correct/TP | Incorrect/FP | FN | Precision/Acc | Recall | F1-score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sentence Level** | 1000 | 728 | - | - | 0.7280 | - | - |
| **Entity Level** | 2297 | 1972 | 311 | 325 | 0.8638 (Acc) | 0.8585 | **0.8611** |
| **Token Level** | 3250 | 2711 | - | - | 0.8438 | 0.8342 | 0.8389 |

---

## 2. numind/NuNER-v2.0 (Baseline)
*Note: Claimed SOTA (20 epochs).*

| Metric Type | Total | Correct/TP | Incorrect/FP | FN | Precision/Acc | Recall | F1-score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sentence Level** | 14422 | 6 | - | - | 0.0004 | - | - |
| **Entity Level** | 33653 | 10242 | 106541 | 23411 | 0.0877 (Acc) | 0.3043 | 0.1362 |
| **Token Level** | 48022 | 13456 | - | - | 0.0425 | 0.2802 | 0.0739 |

> **Note**: NUNER 不使用三元组的格式微调结果。
> WandB: [NuNER_project](https://wandb.ai/jerrylikespython-bnu-hkbu-united-international-college/NuNER_project?nw=nwuserjerrylikespython)

---

## 3. SpaCy Models
*Fine-tuned for 50 epochs.*

### en_core_web_md
| Metric Type | Total | Correct/TP | Incorrect/FP | FN | Precision | Recall | F1-score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sentence Level** | 1000 | 705 | 295 | - | 0.7050 (Acc) | - | - |
| **Entity Level** | 2328 | 1968 | 704 (Err) | - | 0.8512 | 0.8454 | **0.8483** |
| **Token Level** | 3414 | 2811 | 1198 (Err) | - | 0.8253 | 0.8234 | 0.8243 |

### en_core_web_lg
| Metric Type | Total | Correct/TP | Incorrect/FP | FN | Precision | Recall | F1-score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sentence Level** | 1000 | 716 | 284 | - | 0.7160 (Acc) | - | - |
| **Entity Level** | 2328 | 1983 | 687 (Err) | - | 0.8529 | 0.8518 | **0.8524** |
| **Token Level** | 3414 | 2820 | 1225 (Err) | - | 0.8172 | 0.8260 | 0.8216 |

### Electra-base
# 20 epoch
Precision: 0.6522, Recall: 0.5294, F1: 0.5844
# 50 epoch
Precision: 0.7333, Recall: 0.7765, F1: 0.7543
# 200 epoch
Precision: 0.7778, Recall: 0.8235, F1: 0.8000