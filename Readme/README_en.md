# NLP_TEXT_CLASSIFICATION

## Language Selection

- [Chinese README](README_zh.md)

## Project Description

This project contains three main types of model files:

### 1. LLM API Calls

- Supports calling models such as DeepSeek, ChatGPT, and Gemini.
- Experiments are divided into zero-shot and few-shot inference.

### 2. Other Models

- Includes spaCy pretrained pipelines (sm series) and the NuNER model.
- Note: Due to poor fitting results when using position features, alternative algorithms were adopted later.
- `sm0.py` is the latest version, with output format as follows:

  ```
  [[entity0, order, label], [entity1, order, label], ...]
  ```

- SpaCy pipeline details:
  - `sm` series: can directly switch between different models (e.g., md, lg).
  - `trf` series: experiments were terminated due to version compatibility issues (object lifecycle errors).

### 3. LLM Fine-Tuned Models

- Models used:
  - DeepSeek-R1-Distill-Qwen 1.5B / 7B
  - Llama4-17B
- Full fine-tuning experiments have been completed. DeepSeek-R1 Distill 1.5B achieved the highest score during testing, surpassing the original NuNER SOTA.

## LoRA Fine-Tuning Experiment (Ongoing)

- Fine-tuning model: Llama4-17B
- Framework: Axolotl
- Method: INT4 quantized LoRA fine-tuning

## Resource Structure

- `experiment_log/`: Contains historical experimental logs (detailed data will be released after paper publication).
- `model/`: All model source codes.
- `model/FFT/`: Fine-tuned models and inference scripts.

## Released Model

- DeepSeek-R1 1.5B FFT v4 fine-tuned model has been packaged and uploaded.
- Supports Windows terminal inference. Tested on a single 2080Ti GPU without issues.

- [Google Drive - Model Download](https://drive.google.com/file/d/1-L8KtT2USiZtG6eS2rXyOU3rveRQAl-u/view?usp=drive_link)
- [requirements.txt - Environment Dependencies](https://drive.google.com/file/d/1HCBd9aUkgMHEabdsu6Y93Nn7DEhndFkV/view?usp=drive_link)

## Entity Labels

| Label | Meaning |
|------|---------|
| per  | Person |
| org  | Organization |
| gpe  | Geopolitical Entity |
| geo  | Geographical Entity |
| tim  | Time indicator |
| art  | Artifact |
| eve  | Event |
| nat  | Natural Phenomenon |
