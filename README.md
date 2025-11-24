# NLP_TEXT_CLASSIFICATION

📌 **项目 / Project**: 文本分类算法 / Text Classification Algorithm  
本项目旨在开发和优化文本分类相关的算法。  
This project aims to develop and optimize algorithms for text classification.
⚠本项目已经停止更新和维护，在https://github.com/Jerry-poor/NLP_Research.git开启了POS模型二周目
---

## 📚 选择语言 / Choose Language

- [中文 README](./Readme/README_zh.md)
- [English README](./Readme/README_en.md)

---

## ⭐ 全量微调实验已有结果 / Fine-tuning Experiment Results

- 🔧 **中文**：  
  - DeepSeek-R1 1.5B 微调后，在 v5 版本上实体级别达到了 **F1-score 87**，v4 版本为 **F1-score 86**。
  - DeepSeek-R1 7B 微调后达到 **F1-score 86**。
  - v5 版本中已固定训练随机数种子，禁用了 cuDNN 的 benchmark，并启用了 cuDNN 的 deterministic，确保在非 bit 级别的可复现性。
  - 如果本次模型为最终版本（final version），源码将在论文发表后公开。目前 v4 版本的全部源码已公开。如果后续有新模型微调结果超过 v5，将公开 v5 版本源码。

- 🔧 **English**：  
  - After fine-tuning, DeepSeek-R1 1.5B achieved **F1-score 87** on v5 and **F1-score 86** on v4.
  - DeepSeek-R1 7B fine-tuned model achieved **F1-score 86**.
  - In v5, random seeds were fixed, cuDNN benchmark was disabled, and cuDNN deterministic mode was enabled to ensure reproducibility (except for bit-level differences).
  - If this model is confirmed as the final version, the source code will be released after the paper publication. All source code of v4 is already publicly available. If future fine-tuning surpasses v5, v5 source code will also be released.

---

## ❀ 超越 SOTA / Surpassing SOTA

- **中文**：  
  本实验中，我们对 **DeepSeek** 模型进行了微调，在实验数据集上全面超越了现有 SOTA 模型，包括 **NuNER**、**spaCy**（`en_core_web_sm` / `md` / `lg`）以及基于 **CoNLL-2003** 数据集微调的 **ELECTRA**。  
  此外，我们的 **v5 版本**在公开数据集 [abhinavwalia95](https://huggingface.co/datasets/abhinavwalia95) 上也取得了当前最佳（SOTA）性能，进一步验证了模型的泛化能力和实际应用价值。

- **English**：  
  In this study, we fine-tuned the **DeepSeek** model, which outperformed existing state-of-the-art models — including **NuNER**, **spaCy** (`en_core_web_sm` / `md` / `lg`), and **ELECTRA** fine-tuned on **CoNLL-2003** — on our experimental dataset.  
  Furthermore, our **v5 version** achieved **state-of-the-art performance** on the public benchmark dataset [abhinavwalia95](https://huggingface.co/datasets/abhinavwalia95), demonstrating strong generalization and practical effectiveness.


## 📌 更多详细信息 / More Details

- **中文**：[知乎文章 - Zhuanlan.zhihu.com](https://zhuanlan.zhihu.com/p/1892251638514828147)
- **English**：[Zhihu Article (Chinese)](https://zhuanlan.zhihu.com/p/1892251638514828147)
