#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import json
import random
import numpy as np
import pandas as pd
import torch
import wandb
import traceback
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, DataCollatorWithPadding,
    set_seed
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
import argparse
print("ds.py real path:", __file__, flush=True)
#为了确保模型是可以复现的，现在重新固定所有种子和超参数
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark   = False

parser = argparse.ArgumentParser(description='Fine-tune a model using Deepspeed')
parser.add_argument('--deepspeed', type=str, required=True, help='Path to deepspeed configuration file')
parser.add_argument('--train_batch_size', type=int, default=2, help='Training batch size')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
parser.add_argument('--num_train_epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--seed', type=int, default=SEED, help='Random seed for reproducibility')
parser.add_argument('--local_rank', type=int, default=-1, help='Local rank passed from distributed launcher')
args = parser.parse_args()

if args.local_rank == 0:
    api_key = os.environ.get("WANDB_API_KEY")
    print(f"WANDB_API_KEY in env = {api_key}", flush=True)
    if api_key:
        wandb.login(key=api_key, relogin=False)
        wandb.init(project="llm_finetuning", mode="online")
    else:
        print("⚠️ WANDB_API_KEY is not set; disabling wandb", flush=True)
        wandb.init(project="llm_finetuning", mode="disabled")


fixed_prompt = (
    "You are a natural language processing assistant. Please extract all named entities from the input text "
    "and return the results in JSON format. If the same entity appears multiple times, assign an increasing order "
    "starting from 0. "
)

# 从预处理后的 CSV 中读取数据（CSV 使用 latin1 编码）
csv_path = os.path.join(os.path.dirname(__file__), "preprocessed_dataset.csv")
df = pd.read_csv(csv_path, encoding="latin1")
# 划分训练集和测试集
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

def construct_question(row, fixed_prompt):
    """
    构造 question = fixed_prompt + "\nSentence: " + input
    注意：row["input"] 已为预处理后的转义字符串
    """
    return fixed_prompt + "\nSentence: " + row["input"]

def chat(df, fixed_prompt):
    """
    构造 QA 对，每个样本的 question 为上述构造，answer 为 output 列
    """
    qa_pairs = []
    for i, row in df.iterrows():
        question = construct_question(row, fixed_prompt)
        answer = row["output"]
        qa_pairs.append({
            "question": question,
            "answer": answer
        })
    return qa_pairs

qa_pairs = chat(train_df, fixed_prompt)
qa_dataset = Dataset.from_list(qa_pairs)

model_name = "DeepSeek-R1-Distill-Qwen-1.5B_FFT/DeepSeek-R1-Distill-Qwen-1.5B"
print("Loading tokenizer from", model_name, flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
start_token = tokenizer.bos_token if tokenizer.bos_token is not None else "<s>"
end_token = tokenizer.eos_token if tokenizer.eos_token is not None else "</s>"

print("Loading model from", model_name, flush=True)
model = AutoModelForCausalLM.from_pretrained(model_name)


# Tokenization：
def tokenize_function(examples):
    """
    对每个示例构造文本：
      - 构造 question = fixed_prompt + "\nSentence: " + input
      - 为 question 添加起始和结束 token： q_with_tokens = BOS + question + EOS
      - 为 target (output) 添加起始和结束 token： a_with_tokens = BOS + output + EOS
      - 拼接为： full_text = q_with_tokens + "\n" + a_with_tokens
    每个示例独立处理，max_length=128，超出部分直接截断。
    同时将 input_ids 复制一份到 labels 以计算 loss。
    注意：使用 padding="max_length" 固定输出长度为 128。
    """
    texts = []
    for q, a in zip(examples["question"], examples["answer"]):
        q_with_tokens = start_token + q + end_token
        a_with_tokens = start_token + a + end_token
        full_text = q_with_tokens + "\n" + a_with_tokens
        texts.append(full_text)
    encoded = tokenizer(texts, truncation=True, max_length=512, padding="max_length")
    encoded["labels"] = encoded["input_ids"].copy()
    return encoded

print("Starting tokenization...", flush=True)
# 删除原始的 "question" 和 "answer" 列，确保 dataset 中只保留 tokenized 输出
tokenized_dataset = qa_dataset.map(tokenize_function, batched=True, remove_columns=qa_dataset.column_names)
print("Tokenization completed.", flush=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataset = tokenized_dataset  # 每个示例独立

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=20,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # 全局 batch size = num_gpus * 8 * per_device_train_batch_size
    fp16=False,
    deepspeed=args.deepspeed,
    remove_unused_columns=False,
    logging_steps=10,
    eval_strategy="no",
    save_total_limit=3,
    learning_rate=args.learning_rate,
    seed=SEED,
    weight_decay=args.weight_decay
    
)

print("Initializing Trainer...", flush=True)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,  # 训练时不进行评估
    data_collator=data_collator
)
print("Trainer initialized. Starting training...", flush=True)

try:
    trainer.train()
except Exception as e:
    print("Training error:", e, flush=True)
    traceback.print_exc()
    sys.exit(1)

print("Training completed.", flush=True)


final_model_dir = "./final_v5_model"
trainer.save_model(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
print(f"Model and tokenizer saved to {final_model_dir}", flush=True)

