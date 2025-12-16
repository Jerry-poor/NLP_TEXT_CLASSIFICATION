# NLP_TEXT_CLASSIFICATION
 
NER 模型权重已经量化，可以使用ollama直接运行
POS 模型见 https://github.com/Jerry-poor/NLP_Research(此模型只是进行了微调，没有进行对比测评)

## 选择语言 / Choose Language
- [中文 README](./Readme/README_zh.md)
- [English README](./Readme/README_en.md)

## 概览 / Overview
- Text classification & NER experiments across LLM API calls, spaCy baselines, and fine-tuning (DeepSeek-R1, Llama variants, LoRA).
- 大型数据集与模型检查点不随仓库分发，来源见 `dataset/dataset_url.txt`；已配置 `.gitignore` 与 Git LFS 以避免误提交。
- API 与推理侧包含 DeepSeek 的 zero/few-shot 流程与置信度输出（见 `model/API/Deepseek`、`model/API/deepseek-chat`）。

## 目录结构 / Layout
- `dataset/`: 下载链接与处理脚本；CSV/CONLLU/`.spacy` 等大文件已忽略。
- `model/API/`: DeepSeek / ChatGPT / Gemini 等 API 分类器及实验配置、结果脚本。
- `model/FFT/`: DeepSeek-R1 1.5B/7B 微调脚本与配置；检查点与预处理数据不在仓库。
- `model/LoRA/`, `model/AzureFT/`, `model/ELECTRA/`, `model/NuNER/`, `model/Spacy/`: 其他基线与实验。
- `experiment_log/`: 历史实验日志（docx/txt 已忽略）；`Blog/` 含实验笔记。

## 使用提示 / Notes
- 按 `dataset/dataset_url.txt` 下载原始数据，使用对应处理脚本重新生成 spaCy 二进制或预处理 CSV。
- 保持密钥在 `.env`（已忽略）；生成的结果文件、检查点、大型 CSV / spaCy 文件请勿提交，必要时使用 Git LFS。
