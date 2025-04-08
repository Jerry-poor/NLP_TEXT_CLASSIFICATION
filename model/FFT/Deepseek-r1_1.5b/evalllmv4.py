#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split

def construct_question(row, fixed_prompt):
    """
    构造 question：
      fixed_prompt + "\nSentence: " + row["input"]
    """
    return fixed_prompt + "\nSentence: " + row["input"]

def evaluate(model, tokenizer, df_subset, fixed_prompt, start_token, end_token):
    """
    评估流程：
      1) 对于每个样本，从预处理数据中读取 input 和 output；
      2) 使用 construct_question() 构造 question，并添加 BOS/EOS；
      3) 使用模型生成输出（设置 max_new_tokens=64，温度近似为0）；
      4) 生成文本格式：BOS + question + EOS + "\n" + BOS + target + EOS，
         取换行后的第二部分作为生成的 target，去除 BOS/EOS 后解析为 JSON；
      5) 返回详细信息字符串列表，每个样本包括：
           - 原始 sentence（即 input）
           - 真实实体 (Gold Output)
           - 模型生成的输出 (Model Output)
    """
    detailed_results = []
    max_length=512

    for idx, row in df_subset.iterrows():
        # 使用与训练时相同的方式构造 question
        question = construct_question(row, fixed_prompt)
        question_with_tokens = start_token + question + end_token
        gold_output = row["output"]

        # 编码输入，使用较长的 max_length（例如256）以确保不截断问题部分
        inputs = tokenizer(question_with_tokens, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=max_length, do_sample=True, temperature=1e-8)
        full_generated = tokenizer.decode(gen_ids[0], skip_special_tokens=False).strip()

        # 按换行分割，期望格式：BOS + question + EOS + "\n" + BOS + target + EOS
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

        # 构造详细信息字符串，包括原始输入、真实输出和模型生成的输出
        detail = (
            f"Sample {idx+1}:\n"
            f"Sentence (Input): {row['input']}\n"
            f"True Entities (Gold Output): {gold_output}\n"
            f"Model Output: {generated_target}\n"
            "------------------------"
        )
        detailed_results.append(detail)

    return detailed_results

def main():
    # 加载模型和分词器（从最终模型目录加载）
    final_model_dir = "./final_v4_model"
    print("Loading model from", final_model_dir)
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
    # 使用相同随机种子 42 进行 train/test 拆分，并取测试集前200行作为评估集
    _, test_df = train_test_split(df, test_size=0.3, random_state=42)
    df_eval = test_df.iloc[:1000]

    detailed_results = evaluate(model, tokenizer, df_eval, fixed_prompt, start_token, end_token)

    # 将详细评估结果保存到文件
    os.makedirs("ds_1.5b_fft_v4_result", exist_ok=True)
    results_file = os.path.join("ds_1.5b_fft_v4_result", "combined_eval_results.txt")
    with open(results_file, "w", encoding="utf-8") as f:
        for detail in detailed_results:
            f.write(detail + "\n")
    print(f"Combined evaluation results saved to {results_file}")

if __name__ == "__main__":
    main()
