# NLP_TEXT_CLASSIFICATION

## 选择语言 / Choose Language
- [中文 README](README_zh.md)

## 概览
- 支持 LLM API 调用、spaCy/NuNER 基线，以及 DeepSeek-R1 / Llama 等微调与 LoRA 实验。
- 仓库已清理大文件，原始/处理数据与 spaCy 二进制不随库分发，下载链接见 `dataset/dataset_url.txt`；大文件通过 `.gitignore` 与 Git LFS 管理。
- API 侧新增 DeepSeek 的 zero/few-shot 流程与置信度输出（见 `model/API/Deepseek`、`model/API/deepseek-chat`）。

## 模型文件分类
1. **LLM API 调用**：支持 DeepSeek、ChatGPT、Gemini；实验分 zero-shot / few-shot。
2. **其他模型**：spaCy 预训练管道（sm 系列）与 NuNER。`sm0.py` 为最新版，输出格式为 `[[entity, order, label], ...]`。
3. **LLM 微调模型**：DeepSeek-R1-Distill-Qwen 1.5B / 7B、Llama4-17B；已完成全量微调实验。

## LoRA 微调实验（进行中）
- 模型：Llama4-17B
- 框架：Axolotl
- 方式：INT4 量化 LoRA 微调

## 资源结构
- `dataset/`：下载链接与处理脚本；CSV/CONLLU/`.spacy` 等大文件已忽略。
- `model/`：模型代码；`model/FFT/` 为微调脚本，`model/API/` 为各 API 管线与实验结果脚本。
- `experiment_log/`：历史实验日志（docx/txt 已忽略）；`Blog/` 含实验笔记。

## 已发布模型
- DeepSeek-R1 1.5B FFT v4 打包上传，支持 Windows 终端推理（单张 2080Ti 测试通过）。
- [Google Drive - 模型下载](https://drive.google.com/file/d/1-L8KtT2USiZtG6eS2rXyOU3rveRQAl-u/view?usp=drive_link)
- [requirements.txt - 环境依赖](https://drive.google.com/file/d/1HCBd9aUkgMHEabdsu6Y93Nn7DEhndFkV/view?usp=drive_link)

## 抓取类别（实体标签）
| 标签 | 含义 |
|-----|-----|
| per | Person（人名） |
| org | Organization（组织机构） |
| gpe | Geopolitical Entity（政治实体） |
| geo | Geographical Entity（地理实体） |
| tim | Time indicator（时间标识） |
| art | Artifact（人工制品） |
| eve | Event（事件） |
| nat | Natural Phenomenon（自然现象） |
