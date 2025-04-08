#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split

def evaluate_model(model, tokenizer, df_subset, fixed_prompt, start_token, end_token, max_input_length=256, max_new_tokens=64):
    total_samples = 0
    skipped_samples = 0
    sentence_correct = 0

    total_true = 0
    total_pred = 0
    total_tp = 0

    detailed_results = []

    for idx, row in df_subset.iterrows():
        total_samples += 1

        # 构造问题: fixed_prompt + "\nSentence: " + input
        question = fixed_prompt + "\nSentence: " + row["input"]
        question_with_tokens = start_token + question + end_token

        # 检查输入长度
        inputs_for_length = tokenizer(question_with_tokens, return_tensors="pt", truncation=False, padding=False)
        if inputs_for_length["input_ids"].shape[1] >= max_input_length:
            skipped_samples += 1
            continue

        # 编码输入：设置 max_length=max_input_length (例如256) 且使用 padding
        inputs = tokenizer(question_with_tokens, return_tensors="pt", truncation=True, max_length=max_input_length, padding="max_length")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=1e-8)
        full_generated = tokenizer.decode(gen_ids[0], skip_special_tokens=False).strip()

        # 解析生成的文本：
        # 期望格式：BOS + question + EOS + "\n" + BOS + target + EOS
        parts = full_generated.split("\n", 1)
        if len(parts) > 1:
            generated_target = parts[1]
        else:
            generated_target = ""
        if generated_target.startswith(start_token):
            generated_target = generated_target[len(start_token):]
        if generated_target.endswith(end_token):
            generated_target = generated_target[:-len(end_token)]
        generated_target = generated_target.strip()

        # 解析真实标注 (gold output) JSON
        try:
            gold_data = json.loads(row["output"])
        except Exception:
            gold_data = {}
        gold_entities = gold_data.get("entities", [])
        # 转换为元组列表，忽略 order
        gold_tuples = [(e.get("entity", "").strip(), e.get("label", "").strip()) for e in gold_entities]

        # 同样解析模型输出 JSON
        try:
            pred_data = json.loads(generated_target)
        except Exception:
            pred_data = {}
        pred_entities = pred_data.get("entities", [])
        pred_tuples = [(p.get("entity", "").strip(), p.get("label", "").strip()) for p in pred_entities]

        # 使用 Counter 统计
        from collections import Counter
        gold_counter = Counter(gold_tuples)
        pred_counter = Counter(pred_tuples)
        sample_tp = sum(min(pred_counter[k], gold_counter.get(k, 0)) for k in pred_counter)

        total_tp += sample_tp
        total_true += sum(gold_counter.values())
        total_pred += sum(pred_counter.values())

        # 如果预测和真实完全一致，认为该样本句子级预测正确
        if gold_counter == pred_counter:
            sentence_correct += 1

        detail = (
            f"Sample {idx+1}:\n"
            f"Question: {question_with_tokens}\n"
            f"Gold Output: {row['output']}\n"
            f"Generated Target: {generated_target}\n"
            f"Gold Entities: {gold_counter}\n"
            f"Pred Entities: {pred_counter}\n"
            f"Sample TP: {sample_tp}\n"
            "------------------------\n"
        )
        detailed_results.append(detail)

    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall = total_tp / total_true if total_true > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    sentence_acc = sentence_correct / (total_samples - skipped_samples) if (total_samples - skipped_samples) > 0 else 0.0

    metrics = {
        "total_samples": total_samples,
        "skipped_samples": skipped_samples,
        "sentence_accuracy": sentence_acc,
        "total_true_entities": total_true,
        "total_pred_entities": total_pred,
        "total_tp": total_tp,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    return metrics, detailed_results

def main():
    # 从 ./final_model 目录加载模型和分词器
    final_model_dir = "./final_v4_model"
    model = AutoModelForCausalLM.from_pretrained(final_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(final_model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 获取 BOS/EOS 标记
    start_token = tokenizer.bos_token if tokenizer.bos_token is not None else "<s>"
    end_token = tokenizer.eos_token if tokenizer.eos_token is not None else "</s>"

    # 固定 prompt（与训练时一致）
    fixed_prompt = (
        "You are a natural language processing assistant. Please extract all named entities from the input text "
        "and return the results in JSON format. If the same entity appears multiple times, assign an increasing order "
        "starting from 0. "
    )

    # 加载预处理后的 CSV 数据集（latin1 编码）
    csv_path = os.path.join(os.path.dirname(__file__), "preprocessed_dataset.csv")
    df = pd.read_csv(csv_path, encoding="latin1")
    # 使用相同随机种子 42 进行 train/test 拆分，取测试集前200行作为评估集
    _, test_df = train_test_split(df, test_size=0.3, random_state=42)
    df_eval = test_df.iloc[:1000]

    metrics, detailed_results = evaluate_model(model, tokenizer, df_eval, fixed_prompt, start_token, end_token)
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # 将详细评估结果保存到文件
    os.makedirs("ds_1.5b_fft_v4_result", exist_ok=True)
    results_file = os.path.join("ds_1.5b_fft_result", "combined_eval_results.txt")
    with open(results_file, "w", encoding="utf-8") as f:
        for detail in detailed_results:
            f.write(detail + "\n")
    print(f"Combined evaluation results saved to {results_file}")

if __name__ == "__main__":
    main()
