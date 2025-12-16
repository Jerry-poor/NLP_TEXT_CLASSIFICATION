# 高效 LLM 微调与推理框架 (NLP 任务)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-DeepSeek%20%7C%20Qwen-green)](https://github.com/deepseek-ai/DeepSeek-LLM)
[![Task](https://img.shields.io/badge/Task-NER%20%7C%20Text_Classification-orange)]()

> **项目摘要**: 本仓库实现了一个用于 NLP 任务的 LLM 高效适配框架。通过对 **DeepSeek-R1-Distill (1.5B/7B)** 模型进行 **全量微调 (FFT)** 和 **Int8 量化**，本项目在命名实体识别 (NER) 任务上取得了 **SOTA 性能 (F1: 87.1%)**，超越了传统基线 (SpaCy)、判别式模型 (ELECTRA) 和近期研究模型 (NuNER)，同时在消费级硬件上保持了高效推理。

## 🌐 语言
- [English README](../README.md) (英文版 - Main)

## 📚 技术实现博客 (推荐阅读)
> 我在技术博客中记录了完整的思考过程和实现细节。**强烈推荐** 阅读以深入理解 `DeepSeek` 微调全流程。
> * **[使用LLM进行NER任务：Deepseek蒸馏小模型微调实战 (SOTA)](https://zhuanlan.zhihu.com/p/1892251638514828147)**
> * **[微调 Deepseek-r1:7b 模型进行 NER 任务：从 0 到 1](https://zhuanlan.zhihu.com/p/1895169190219974297)**

---

---

## 🔬 研究方向与方法论

### 1. 命名实体识别 (NER) - *核心方向*
利用生成式推理能力提取复杂实体。

#### **方法论:**
* **LLM 全量微调 (FFT)**: 使用监督微调成功适配 **DeepSeek-R1-Distill-Qwen** (1.5B & 7B)。
* **基线对比**: 对比 **SpaCy** (sm/md/lg 管道)、**ELECTRA** (Transformer Encoder) 和 **NuNER** (学术界 SOTA)。
* **设计决策与约束**:
    * *SpaCy `trf`*: 调研了基于 Transformer 的管道，但因版本兼容性影响复现而被排除。
    * *Logprobs 约束*: 探索了基于 Token 对数概率的约束解码；由于当前 API 限制无法精确获取细粒度概率而放弃。
* **量化 (进行中)**: 正在尝试 Llama-4-17B (INT4) 的 **QLoRA** 和 **Axolotl** 实验，测试极限模型压缩。

#### **🏆 实验结果 (基准测试)**

**主要发现:**
1.  **统治级性能**: 微调后的 **DeepSeek-R1-Distill (1.5B)** 取得了 **实体级 F1 87.1%**，显著优于 NuNER 基线 (**Token 级 F1 ~79%**) 和 ELECTRA-Base (**F1 80.0%**)。
    > *注*: 在严格的实体级评估 (困难模式) 下达到 87%，对比 Token 级 (简单模式) 的 79%，证明了微调生成式模型更优越的结构理解能力。
2.  **小模型 vs 大模型**: 1.5B 模型 (确定性 FFT) 优于更大的 **DeepSeek-7B** 量化模型 (86.1%)，证明了精细微调比单纯扩大规模具有更高的 ROI。
3.  **高效性**: 在消费级硬件 (RTX 2080Ti) 上达到 SOTA 性能，推理延迟极低。

| 模型 | 规模 | Precision | Recall | F1-Score | 评估层级 | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DeepSeek-R1 (v5)** | **1.5B** | **0.8715** | **0.8707** | **0.8711** | **实体级 (严格)** | **最佳性能**, 确定性训练 |
| DeepSeek-R1 | 7B | 0.8638 | 0.8585 | 0.8611 | 实体级 (严格) | Int8 量化推理 |
| SpaCy (model: lg) | - | 0.8529 | 0.8518 | 0.8524 | 实体级 (严格) | 工业界基线 |
| **ELECTRA-Base** | - | 0.7778 | 0.8235 | 0.8000 | 实体级 (严格) | Encoder-only 基线 |
| **NuNER** | - | 0.7778 | 0.8235 | 0.8000 | **Token 级 (宽松)** | 学术界基线 (20 epoch) |
| SpaCy (model: md) | - | 0.8512 | 0.8454 | 0.8483 | 实体级 (严格) | 标准基线 |

> **评估协议**:
> * **实体级 (严格)**: 要求边界和实体类型完全匹配 (用于 DeepSeek, SpaCy, ELECTRA)。
> * **Token 级 (宽松)**: 逐字符或逐 Token 计算指标 (用于 NuNER 基线)。
> * **结果解读**: DeepSeek-R1 的优势比数字显示的更显著，因为它是在更严格的评估指标上取得的。

### 2. 文本分类 - *新型置信度管线*
* **Method 1 (已实现)**: 开发了基于 LLM 生成置信度分数的管线，在 **Zero-shot** 和 **Few-shot** 场景下非常有效。
* **Method 2 (服务商不支持)**: 调研了通过 Logprobs 进行外部控制；确认了当前 API 在细粒度概率获取上的限制。
* *位置*: `model/API/method1`

## 📈 文本分类实验结果 (详细版)
> **[点击此处查看完整的 API 分类实验报告](result/Text-Classifiaciton_API.md)**

如需查看详细实现代码和原始结果文件，请访问 **[`model/API`](model/API)** 目录。

### 3. POS (词性标注)
* *状态*: 已迁移至独立研究仓库: [NLP_Research](https://github.com/Jerry-poor/NLP_Research.git)。

---

## ⚡ 关键技术栈

本项目使用业界标准的 ML 工程栈以确保可扩展性和复现性。

| 类别 | 技术 |
| :--- | :--- |
| **核心框架** | `PyTorch` • `HuggingFace Transformers` • `Datasets` |
| **训练 & 运维** | `DeepSpeed` (ZeRO-2/3) • `Axolotl` • `WandB` (实验追踪) |
| **HPC 基础设施** | **NVIDIA A100 / A800 Clusters** (分布式训练) • `Slurm` |
| **数据工程** | `Pandas` • `Scikit-learn` • 自定义数据管线 |
| **复现性** | 确定性 cuDNN • 固定随机种子 (42) |

---

## 📦 模型库与工件 (Model Zoo)

我提供了预训练权重和量化模型以促进复现。

* **DeepSeek-R1 1.5B FFT (v4)**
    * *描述*: 全量微调模型，打包用于推理。
    * *兼容性*: 支持 Windows 终端推理；在单张 RTX 2080Ti 上测试通过。
    * **[Google Drive 下载链接](https://drive.google.com/file/d/1-L8KtT2USiZtG6eS2rXyOU3rveRQAl-u/view?usp=drive_link)**
    * **[环境依赖 (requirements.txt)](https://drive.google.com/file/d/1HCBd9aUkgMHEabdsu6Y93Nn7DEhndFkV/view?usp=drive_link)**

## 🛠️ 项目结构

```bash
.
├── dataset/                  # 数据处理脚本与加载器
├── experiment_log/           # 训练日志 (原始数据保留用于发表)
├── model/
│   ├── API/                  # 推理管线 (文本分类)
│   ├── FFT/                  # 全量微调实现
│   │   ├── Deepseek-r1_1.5b/ # SOTA 模型脚本
│   │   └── Deepseek-r1_7b/   # 量化模型脚本
│   └── Baselines/            # SpaCy, ELECTRA & NuNER 基准测试脚本
└── axolotl_config.yaml       # LoRA 实验配置
```

## 🏷️ 实体模式 (Entity Schema)
模型经过训练可识别以下细粒度实体类型：

| 标签 | 定义 | 标签 | 定义 |
| --- | --- | --- | --- |
| **PER** | 人名 (Person) | **TIM** | 时间标识 (Time Indicator) |
| **ORG** | 组织机构 (Organization) | **ART** | 人工制品 (Artifacts) |
| **GPE** | 政治实体 (Geopolitical Entity) | **EVE** | 事件 (Events) |
| **GEO** | 地理实体 (Geographical Entity) | **NAT** | 自然现象 (Natural Phenomena) |

---


---

## 📂 文件详细说明 / File Descriptions

### 1. 数据处理 (`dataset/`)
- `dataset_url.txt`: 原始数据的下载链接集合。
- `label.py`, `uni.py`: 用于检查数据集中唯一实体标签的辅助脚本。
- `dataset0/`: 包含部分示例或预处理数据。

### 2. NER 实验 (`model/FFT/`, `model/*/`)
#### DeepSeek-R1 1.5B (`model/FFT/Deepseek-r1_1.5b/`)
- `ds_v4.py` / `ds_v5.py`: 微调训练脚本。**v5** 为最新版，启用了确定性训练设置。
- `eval_v4.py`, `eval_v5.py`: 对应的模型评估脚本。
- `ds_1.5b_v4_train_parse.py`: 训练日志解析工具。
- `combined_eval_results_*.txt`: 汇总的评估指标结果。

#### DeepSeek-R1 7B (`model/FFT/Deepseek-r1_7b/`)
- `ds_7b.py`: 7B 模型的微调训练脚本。
- `evalllm7b.py`: 7B 模型的评估推理脚本。
- `ds_7b_output.csv`: 7B 模型的推理输出结果。

#### 基线模型 (Baselines)
- `model/Spacy/sm0.py`: SpaCy (sm/md/lg) 基线测试脚本。
- `model/NuNER/NuNER.py`: NuNER (SOTA) 基线测试脚本。
- `model/ELECTRA/ELE.py`: ELECTRA 模型基线测试脚本。

### 3. API 文本分类 (`model/API/method1/`)
#### 核心脚本 (`scripts/`)
- `main.py`: 主程序入口。支持 Zero-shot, One-shot, Few-shot 分类任务的调度。
- `llm_classifier_utils.py`: 核心工具库。封装了 DeepSeek/OpenAI 等 API 调用、Prompt 构建、置信度解析及准确率计算。
- `zero/one/few_shot_classifier_with_confidence.py`: 各分类模式的具体实现逻辑。
- `ag_news_multi_level_eval.py`: 针对 AG News 数据集的多层级分类评估专用脚本。

#### 资源与结果
- `results/`: 存放所有 API 分类实验生成的 CSV 结果文件。
- `datasets/`: 存放 API 实验使用的特定数据集。
