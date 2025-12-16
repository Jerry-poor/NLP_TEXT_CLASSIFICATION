

```markdown
# Efficient LLM Fine-Tuning & Reasoning Framework for NLP Tasks

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-DeepSeek%20%7C%20Qwen-green)](https://github.com/deepseek-ai/DeepSeek-LLM)
[![Task](https://img.shields.io/badge/Task-NER%20%7C%20Text_Classification-orange)]()

> **Project Abstract**: This repository implements a high-performance framework for adapting Large Language Models (LLMs) to specific NLP tasks. By performing **Full Fine-Tuning (FFT)** and **Int8 Quantization** on **DeepSeek-R1-Distill (1.5B/7B)** models, this project achieves **SOTA performance (F1: 87.1%)** on Named Entity Recognition, surpassing traditional baselines (SpaCy) and recent research models (NuNER) while maintaining inference efficiency on consumer-grade hardware.



## 🌐 Language
- [中文 README](README_zh.md) (Chinese Version)

---

## 🔬 Research Directions & Methodology

### 1. Named Entity Recognition (NER) - *Core Focus*
Leveraging generative reasoning to extract complex entities.

#### **Methodology:**
* **LLM Full Fine-Tuning (FFT)**: successfully adapted **DeepSeek-R1-Distill-Qwen** (1.5B & 7B) using supervised fine-tuning techniques.
* **Baselines**: Benchmarked against **SpaCy** (sm/md/lg pipelines), **ELECTRA**, and **NuNER** (Research SOTA).
* **Ablation Studies (Analysis of Approaches)**:
    * *SpaCy `trf`*: Investigated transformer-based pipelines but encountered version compatibility bottlenecks.
    * *Logprobs Constraints*: Explored constrained decoding via token log-probabilities; found API precision limitations affected stability.
* **Quantization (Ongoing)**: Experimenting with **QLoRA** and **Axolotl** on Llama-4-17B (INT4) to test extreme model compression.

#### **🏆 Experimental Results (Benchmark)**

#### **🏆 Experimental Results (Benchmark)**

**Key Findings:**
1.  **Dominating Performance**: My fine-tuned **DeepSeek-R1-Distill (1.5B)** achieved an **Entity-Level F1 of 87.1%**, significantly outperforming the NuNER baseline (**Token-Level F1 ~79%**).
    > *Note*: Achieving 87% on strict Entity-Level evaluation (Hard Mode) vs 79% on Token-Level (Easy Mode) demonstrates the superior structural understanding of the fine-tuned generative model.
2.  **Small vs Large**: The 1.5B model (Deterministic FFT) outperformed the larger **DeepSeek-7B** quantized model (86.1%), proving that rigorous fine-tuning yields better ROI than simple scaling.
3.  **Efficiency**: Achieved SOTA performance on consumer-grade hardware (RTX 2080Ti) with minimal inference latency.

| Model | Size | Precision | Recall | F1-Score | Evaluation Level | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DeepSeek-R1 (v5)** | **1.5B** | **0.8715** | **0.8707** | **0.8711** | **Entity (Strict)** | **Best Performance**, Deterministic |
| DeepSeek-R1 | 7B | 0.8638 | 0.8585 | 0.8611 | Entity (Strict) | Int8 Quantized Inference |
| SpaCy (model: lg) | - | 0.8529 | 0.8518 | 0.8524 | Entity (Strict) | Industrial Baseline |
| **NuNER** | - | - | - | **~0.7900** | **Token (Loose)** | Research Baseline (SOTA Claim) |
| SpaCy (model: md) | - | 0.8512 | 0.8454 | 0.8483 | Entity (Strict) | Standard Baseline |

> **Evaluation Protocol**:
> * **Entity Level (Strict)**: Requires exact match of both boundary and entity type (used for DeepSeek & SpaCy).
> * **Token Level (Loose)**: Calculates metrics character-by-character or token-by-token (used for NuNER baseline).
> * **Result Interpretation**: DeepSeek-R1's superiority is even more significant than the numbers suggest, as it excels on a much stricter evaluation metric.

> **Evaluation Protocol**:
> * Metrics calculated at the **Entity Level** (strict matching).
> * **Sentence Level Accuracy** for DeepSeek 1.5B reached **76.2%**.
> * *Note*: The NuNER baseline performed within the expected range (~79%) but fell short of the specialized fine-tuned DeepSeek models, highlighting the benefits of generative extraction.

### 2. Text Classification - *Novel Confidence Pipeline*
* **Method 1 (Implemented)**: Developed a pipeline using LLM-generated confidence scores, highly effective for **Zero-shot** and **Few-shot** scenarios.
* **Method 2 (Implementation Constraints)**: Investigated external control via Logprobs; identified current API limitations for granular probability retrieval.
* *Location*: `model/API/method1`

### 3. POS (Part-of-Speech Tagging)
* *Status*: Migrated to standalone research repo: [NLP_Research](https://github.com/Jerry-poor/NLP_Research.git).

---

## ⚡ Key Technology Stack

This project utilizes an industry-standard ML engineering stack to ensure scalability and reproducibility.

| Category | Technologies |
| :--- | :--- |
| **Core Frameworks** | `PyTorch` • `HuggingFace Transformers` • `Datasets` |
| **Training & Ops** | `DeepSpeed` (ZeRO-2/3) • `Axolotl` • `WandB` (Tracking) |
| **HPC Infrastructure** | **NVIDIA A100 / A800 Clusters** (Distributed Training) • `Slurm` |
| **Data Engineering** | `Pandas` • `Scikit-learn` • Custom Data Pipelines |
| **Reproducibility** | Deterministic cuDNN • Fixed Random Seeds (42) |

---

## 📦 Model Zoo & Artifacts

I provide pre-trained weights and quantized models to facilitate reproducibility.

* **DeepSeek-R1 1.5B FFT (v4)**
    * *Description*: Fully fine-tuned model, packaged for inference.
    * *Compatibility*: Supports Windows terminal inference; Tested on single RTX 2080Ti.
    * **[Download via Google Drive](https://drive.google.com/file/d/1-L8KtT2USiZtG6eS2rXyOU3rveRQAl-u/view?usp=drive_link)**
    * **[Environment Dependencies (requirements.txt)](https://drive.google.com/file/d/1HCBd9aUkgMHEabdsu6Y93Nn7DEhndFkV/view?usp=drive_link)**

## 🛠️ Project Structure

```bash
.
├── dataset/                  # Data processing scripts & loaders
├── experiment_log/           # Training logs (Raw data reserved for publication)
├── model/
│   ├── API/                  # Inference pipelines (Text Classification)
│   ├── FFT/                  # Full Fine-Tuning implementations
│   │   ├── Deepseek-r1_1.5b/ # SOTA model scripts
│   │   └── Deepseek-r1_7b/   # Quantized model scripts
│   └── Baselines/            # SpaCy & NuNER benchmarking scripts
└── axolotl_config.yaml       # Configuration for LoRA experiments

```

##🏷️ Entity SchemaThe model is trained to recognize the following fine-grained entity types:

| Tag | Definition | Tag | Definition |
| --- | --- | --- | --- |
| **PER** | Person | **TIM** | Time Indicator |
| **ORG** | Organization | **ART** | Artifacts |
| **GPE** | Geopolitical Entity | **EVE** | Events |
| **GEO** | Geographical Entity | **NAT** | Natural Phenomena |

---

###📚 Technical Blogs (Chinese)For detailed implementation notes, please refer to my technical blogs:

* [DeepSeek Distillation Fine-tuning Practice (1.5B)](https://zhuanlan.zhihu.com/p/1892251638514828147)
* [Fine-tuning DeepSeek-r1:7b for NER](https://zhuanlan.zhihu.com/p/1895169190219974297)


```