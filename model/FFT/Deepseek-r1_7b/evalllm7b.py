#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.model_selection import train_test_split
import argparse
import traceback
import time
# 固定 prompt（与训练时保持一致）
FIXED_PROMPT = (
    "You are a natural language processing assistant. Please extract all named entities from the input text "
    "and return the results in JSON format. If the same entity appears multiple times, assign an increasing order "
    "starting from 0."
)

def compute_long(row, tokenizer, fixed_prompt, start_token, end_token):
    """
    对每个样本构造 QA 对：
      - question = fixed_prompt + "\nSentence: " + row["input"]
      - answer = row["output"]
      - 构造完整文本：BOS + question + EOS + "\n" + BOS + answer + EOS
    若 token 数大于 256 返回 1，否则返回 0。
    """
    question = fixed_prompt + "\nSentence: " + row["input"]
    answer = row["output"]
    full_text = start_token + question + end_token + "\n" + start_token + answer + end_token
    tokens = tokenizer(full_text, add_special_tokens=False)
    return 1 if len(tokens["input_ids"]) > 256 else 0

def generate_model_output(question, model, tokenizer, device, start_token, end_token):
    """
    以输入 question（由 FIXED_PROMPT 与原始 input 拼接而成），
    在两端添加 BOS/EOS 后进行词化（固定 max_length 为256），
    调用模型进行推理，解码后直接返回生成文本（包括 BOS/EOS 等特殊符号，不做任何解析）。
    """
    full_text = start_token + question + end_token
    encoded = tokenizer(
        full_text,
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt"
    )
    # 将所有 tensor 移动到模型所在设备
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.inference_mode():
        gen_ids = model.generate(**encoded, do_sample=True, temperature=1e-8)
    # 解码后直接返回，不做任何处理
    generated_text = tokenizer.decode(gen_ids[0], skip_special_tokens=False)
    return generated_text

def main():
    parser = argparse.ArgumentParser(
        description="构造 QA 对、计算 token 数，拆分训练与测试集（过滤 long==0 的样本），对测试集推理，并保存结果CSV。"
    )
    # 定义命令行参数，同时兼容分布式启动传入的 --local_rank 与 --final_model_dir
    parser.add_argument("--model_dir", type=str, default="./model_dir", help="模型目录路径")
    parser.add_argument("--final_model_dir", type=str, default=None, help="兼容旧参数名，优先使用此参数")
    parser.add_argument("--csv_path", type=str, default="preprocessed_dataset.csv", help="预处理后的 CSV 数据集路径")
    parser.add_argument("--local_rank", type=int, default=-1, help="分布式运行时每个进程的 local rank")
    args = parser.parse_args()

    # 若使用 --final_model_dir 则优先使用该参数作为模型目录
    model_path = args.final_model_dir if args.final_model_dir is not None else args.model_dir

    # 如果是在分布式环境中启动（local_rank != -1），初始化分布式进程组
    if args.local_rank != -1:
        torch.distributed.init_process_group(backend="nccl")
        rank = args.local_rank
    else:
        rank = 0

    # 使用 BitsAndBytesConfig 加载 8-bit 量化模型，避免 load_in_8bit 参数的弃用警告
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 关键修改：明确指定 device_map 而非使用 "auto"，以避免生成 DTensor 对象
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": rank},
        low_cpu_mem_usage=True
    )
    # 获取模型所在设备（此时应为 "cuda:<rank>"）
    device = next(model.parameters()).device

    # 设置 BOS 与 EOS 符号
    start_token = tokenizer.bos_token if tokenizer.bos_token is not None else "<s>"
    end_token = tokenizer.eos_token if tokenizer.eos_token is not None else "</s>"

    # 读取 CSV 数据集（要求至少包含 "input" 与 "output" 列）；若无 "sentence" 列则复制 "input"
    df = pd.read_csv(args.csv_path, encoding="latin1")
    if "sentence" not in df.columns:
        df["sentence"] = df["input"]

    print("原始 DF 行数:", df.shape[0])

    # 对整个 DF 计算新列 long（判断 token 数是否超过 256）
    print("正在计算每个样本的 token 数是否超过 256 ...")
    df["long"] = df.apply(lambda row: compute_long(row, tokenizer, FIXED_PROMPT, start_token, end_token), axis=1)

    # 训练与测试拆分，假设测试集占 30%
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    train_long0 = train_df[train_df["long"] == 0]
    test_long0 = test_df[test_df["long"] == 0]

    print("训练集 long==0 的行数:", train_long0.shape[0])
    print("测试集 long==0 的行数:", test_long0.shape[0])

    # 构造 new_df：从测试集中筛选出 long==0 的行，仅保留 "sentence", "input", "output" 三列
    new_df = test_long0[["sentence", "input", "output"]].iloc[:1000].copy()

    # 对 new_df 中的每一行构造 question，并调用模型生成输出，保存到新列 "model_output"
    model_outputs = []
    total_samples = len(new_df)
    print("Inference...")
    for idx, row in new_df.iterrows():
        question = FIXED_PROMPT + "\nSentence: " + row["input"]
        start_time = time.time()
        output_text = generate_model_output(question, model, tokenizer, device, start_token, end_token)
        elapsed_time = time.time() - start_time
        token_count = len(tokenizer(output_text, add_special_tokens=False)["input_ids"])
        print(f" cost time {elapsed_time:.2f}s")
        model_outputs.append(output_text)
    new_df["model_output"] = model_outputs

    # 构造 CSV 文件，仅包含 "sentence", "output", "model_output" 三列
    output_df = new_df[["sentence", "output", "model_output"]]
    output_path = "evaluation_output.csv"
    output_df.to_csv(output_path, index=False, encoding="utf-8")
    print("CSV 文件已保存至:", output_path)

    # 分布式环境下，确保退出前销毁进程组，避免资源泄露
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("程序执行出错:", e)
        traceback.print_exc()
