#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
import argparse

def evaluate(model, tokenizer, df_subset, fixed_prompt, start_token, end_token):
    """
    评估流程：
      1) 对于每个样本，从 CSV 中读取 input 和 output；
      2) 构造 question = fixed_prompt + "\nSentence: " + input，并添加 BOS/EOS；
      3) 模型生成时使用 max_new_tokens=32，温度设为 0；
      4) 生成文本格式：BOS + question + EOS + "\n" + BOS + target + EOS，
         取换行后的第二部分作为生成的 target，去除 BOS/EOS 后解析为 JSON；
      5) 将生成的 target 与真实 output（解析后的 "entities"）对比，计算各项指标。
    """
    total_samples = 0
    sentence_correct = 0

    entity_tp = 0
    entity_fp = 0
    entity_fn = 0

    token_precision_sum = 0
    token_recall_sum = 0
    token_f1_sum = 0

    detailed_results = []

    for idx, row in df_subset.iterrows():
        total_samples += 1
        question = fixed_prompt + "\nSentence: " + row["input"]
        question_with_tokens = start_token + question + end_token
        gold_output = row["output"]

        inputs = tokenizer(question_with_tokens, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            # 使用 max_new_tokens 指定生成新 token 数量
            gen_ids = model.generate(**inputs, max_new_tokens=32, temperature=0.0)
        full_generated = tokenizer.decode(gen_ids[0], skip_special_tokens=False).strip()

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

        try:
            gold_data = json.loads(gold_output)
        except Exception:
            gold_data = {}
        gold_entities = gold_data.get("entities", [])
        set_true = set((e.get("entity", ""), e.get("order", 0), e.get("label", "")) for e in gold_entities)

        def parse_json(text):
            try:
                data = json.loads(text)
                if isinstance(data, dict) and "entities" in data:
                    return data["entities"]
                return []
            except Exception:
                return []
        pred_list = parse_json(generated_target)
        set_pred = set((p.get("entity", ""), p.get("order", 0), p.get("label", "")) for p in pred_list)

        if set_true == set_pred:
            sentence_correct += 1

        tp_ = len(set_true & set_pred)
        fp_ = len(set_pred - set_true)
        fn_ = len(set_true - set_pred)
        entity_tp += tp_
        entity_fp += fp_
        entity_fn += fn_

        true_tokens = json.dumps(gold_entities, ensure_ascii=False).split()
        pred_tokens = generated_target.split()
        if len(true_tokens) > 0 and len(pred_tokens) > 0:
            common_count = len(set(true_tokens) & set(pred_tokens))
            prec = common_count / len(pred_tokens)
            rec = common_count / len(true_tokens)
            f1_ = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        else:
            prec = rec = f1_ = 0
        token_precision_sum += prec
        token_recall_sum += rec
        token_f1_sum += f1_

        detail = (
            f"Sample {total_samples}:\n"
            f"Question: {question_with_tokens}\n"
            f"Gold Output: {gold_output}\n"
            f"Generated Target (after removing tokens): {generated_target}\n"
            f"TP={tp_} FP={fp_} FN={fn_}\n"
            "------------------------\n"
        )
        detailed_results.append(detail)

    def safe_div(a, b):
        return a / b if b else 0

    sentence_acc = safe_div(sentence_correct, total_samples)
    def precision(tp, fp):
        return safe_div(tp, tp + fp)
    def recall(tp, fn):
        return safe_div(tp, tp + fn)
    def f1(p, r):
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    p_e = precision(entity_tp, entity_fp)
    r_e = recall(entity_tp, entity_fn)
    f_e = f1(p_e, r_e)

    token_p = safe_div(token_precision_sum, total_samples)
    token_r = safe_div(token_recall_sum, total_samples)
    token_f = safe_div(token_f1_sum, total_samples)

    eval_metrics = {
        "sentence_accuracy": round(sentence_acc, 4),
        "entity_precision": round(p_e, 4),
        "entity_recall": round(r_e, 4),
        "entity_f1": round(f_e, 4),
        "token_precision": round(token_p, 4),
        "token_recall": round(token_r, 4),
        "token_f1": round(token_f, 4)
    }
    return eval_metrics, detailed_results

def main():
    # 加载模型与分词器，从保存的目录加载
    final_model_dir = "./final_model"
    print("Loading model from", final_model_dir)
    model = AutoModelForCausalLM.from_pretrained(final_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(final_model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 获取 BOS/EOS 标记
    start_token = tokenizer.bos_token if tokenizer.bos_token is not None else "<s>"
    end_token = tokenizer.eos_token if tokenizer.eos_token is not None else "</s>"

    # 固定 prompt，与训练时一致
    fixed_prompt = (
        "You are a natural language processing assistant. Please extract all named entities from the input text "
        "and return the results in JSON format. If the same entity appears multiple times, assign an increasing order "
        "starting from 0. "
    )

    # 加载预处理后的 CSV 数据集（latin1 编码）
    csv_path = os.path.join(os.path.dirname(__file__), "preprocessed_dataset.csv")
    df = pd.read_csv(csv_path, encoding="latin1")
    # 使用相同的随机种子 42 进行 train/test 拆分，取测试集前200行作为评估集
    _, test_df = train_test_split(df, test_size=0.3, random_state=42)
    df_eval = test_df.iloc[:200]

    eval_metrics, detailed_results = evaluate(model, tokenizer, df_eval, fixed_prompt, start_token, end_token)
    print("Evaluation Metrics:")
    for key, value in eval_metrics.items():
        print(f"{key}: {value}")

    # 保存详细评估结果到文件
    os.makedirs("ds_1.5b_fft_result", exist_ok=True)
    results_file = os.path.join("ds_1.5b_fft_result", "eval_results.txt")
    with open(results_file, "w", encoding="latin1") as f:
        for line in detailed_results:
            f.write(line + "\n")
    print(f"Detailed evaluation results saved to {results_file}")

if __name__ == "__main__":
    main()
