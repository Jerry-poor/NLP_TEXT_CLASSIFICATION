# NLP_TEXT_CLASSIFICATION

## 选择语言 / Choose Language

- [中文 README](README_zh.md)

## 文件说明

本项目的模型文件主要分为三类：

### 1. LLM API 调用

- 支持调用 DeepSeek、ChatGPT、Gemini 等模型。
- 实验分为 zero-shot 和 few-shot 推理。

### 2. 其他模型

- 包括使用 spaCy 预训练管道（sm 系列）和 NuNER 模型。
- 注意：由于使用 position 特征时模型拟合效果不佳，后续更换了算法。
- `sm0.py` 为最新版本，输出格式如下：

  ```
  [[entity0, order, label], [entity1, order, label], ...]
  ```

- spaCy 管道说明：
  - `sm` 系列：可直接切换文本处理模型（如 md、lg）。
  - `trf` 系列：由于版本兼容问题导致实验中止（对象生命周期异常）。

### 3. LLM 微调模型

- 使用模型：
  - DeepSeek-R1-Distill-Qwen 1.5B / 7B
  - Llama4-17B
- 已完成全量微调实验。DeepSeek-R1 Distill 1.5B 版本在测试中取得最高分数，超越原 NuNER SOTA 表现。

## LoRA 微调实验（进行中）

- 微调模型：Llama4-17B
- 框架：Axolotl
- 微调方式：INT4量化 LoRA 微调

## 资源结构说明

- `experiment_log/`：包含历史实验日志（详细实验数据将在论文发表后公开）。
- `model/`：存放所有模型代码。
- `model/FFT/`：存放微调后的模型及推理脚本。

## 已发布模型

- DeepSeek-R1 1.5B FFT v4 微调模型已打包上传。
- 支持 Windows 终端推理，测试硬件为单张 2080Ti，无推理压力。

- [Google Drive - 模型下载](https://drive.google.com/file/d/1-L8KtT2USiZtG6eS2rXyOU3rveRQAl-u/view?usp=drive_link)
- [requirements.txt - 环境依赖文件](https://drive.google.com/file/d/1HCBd9aUkgMHEabdsu6Y93Nn7DEhndFkV/view?usp=drive_link)

## 抓取类别（实体标签）

| 标签 | 含义 |
|-----|-----|
| per | Person（人名） |
| org | Organization（组织机构） |
| gpe | Geopolitical Entity（政治实体） |
| geo | Geographical Entity（地理实体） |
| tim | Time indicator（时间标志） |
| art | Artifact（人工制品） |
| eve | Event（事件） |
| nat | Natural Phenomenon（自然现象） |
