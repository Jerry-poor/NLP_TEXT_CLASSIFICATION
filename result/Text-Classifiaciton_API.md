# Text Classification API Performance Report

> **Generated on**: 2026-01-26
> **Source**: `model/API/result/model_comparison_summary_v2.csv`
> **Note**: **Hit Rate** refers to the **Hierarchical Hit Rate** (probability of correctly predicting all levels for a single sample).

## 1. Dataset: Lib (Library Science) - 3 Levels
*Hierarchy*: Level 1 -> Level 2 -> Level 3

| Model | Level 1 Acc | Level 2 Acc | Level 3 Acc | **Hit Rate** |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen3-235B** | 58.60% | 64.80% | 47.00% | **17.85%** |
| **GPT-5.2** | 58.00% | 66.40% | 49.80% | **19.18%** |
| **DeepSeek-Chat** | 62.20% | 61.80% | 45.20% | **17.37%** |

### Detailed Metrics (Latencies in ms)
| Model | L1 Top-3 | L1 Latency | L2 Top-3 | L2 Latency | L3 Top-3 | L3 Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen3-235B | 88.60% | 11079 | 84.20% | 10742 | 69.20% | 10534 |
| GPT-5.2 | 90.40% | 10471 | 85.80% | 11444 | 74.00% | 10587 |
| DeepSeek-Chat | 91.40% | 11705 | 83.60% | 13186 | 68.60% | 11686 |

---

## 2. Dataset: AG News - 3 Levels
*Hierarchy*: Generated DDC-like structure

| Model | Level 1 Acc | Level 2 Acc | Level 3 Acc | **Hit Rate** |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen3-235B** | 61.20% | 61.40% | 47.40% | **17.81%** |
| **GPT-5.2** | 63.80% | 61.40% | 52.00% | **20.37%** |
| **DeepSeek-Chat** | 64.00% | 55.20% | 48.20% | **17.03%** |

### Detailed Metrics
| Model | L1 Top-3 | L1 Latency | L2 Top-3 | L2 Latency | L3 Top-3 | L3 Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen3-235B | 87.80% | 11568 | 86.00% | 11208 | 60.00% | 10972 |
| GPT-5.2 | 89.00% | 10636 | 88.40% | 10426 | 74.60% | 10565 |
| DeepSeek-Chat | 90.40% | 14059 | 85.80% | 12460 | 62.20% | 12633 |

---

## 3. Dataset: WOS (Web of Science) - 2 Levels
*Hierarchy*: Level 1 -> Level 2

| Model | Level 1 Acc | Level 2 Acc | **Hit Rate** |
| :--- | :--- | :--- | :--- |
| **Qwen3-235B** | 44.80% | 76.60% | **34.32%** |
| **GPT-5.2** | 40.20% | 78.60% | **31.60%** |
| **DeepSeek-Chat** | 45.40% | 71.20% | **32.32%** |

### Detailed Metrics
| Model | L1 Top-3 | L1 Latency | L2 Top-3 | L2 Latency |
| :--- | :--- | :--- | :--- | :--- |
| Qwen3-235B | 88.00% | 11797 | 92.00% | 18483 |
| GPT-5.2 | 86.20% | 10347 | 96.00% | 10311 |
| DeepSeek-Chat | 93.20% | 12066 | 94.00% | 12397 |
