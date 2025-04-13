#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import pandas as pd
import torch
import wandb
import traceback
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
from sklearn.model_selection import train_test_split
import argparse
import bitsandbytes as bnb
from transformers import Trainer 

print("ds.py real path:", __file__, flush=True)


parser = argparse.ArgumentParser(description='Fine-tune a model using Deepspeed')
parser.add_argument('--deepspeed', type=str, required=True, help='Path to deepspeed configuration file')
parser.add_argument('--train_batch_size', type=int, default=2, help='Training batch size')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
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


# prompt
fixed_prompt = (
    "You are a natural language processing assistant. Please extract all named entities from the input text "
    "and return the results in JSON format. If the same entity appears multiple times, assign an increasing order "
    "starting from 0. "
)

# QA 对构造
csv_path = os.path.join(os.path.dirname(__file__), "preprocessed_dataset.csv")
df = pd.read_csv(csv_path, encoding="latin1")
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

def construct_question(row, fixed_prompt):
    """
    构造 question = fixed_prompt + "\nSentence: " + input
    注意：row["input"] 已为预处理后的转义字符串
    """
    q = fixed_prompt + "\nSentence: " + row["input"]
    return q

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

#加载模型与分词器，并获取 BOS/EOS 标记

model_name = "DeepSeek-R1-Distill-Qwen-7B/DeepSeek-R1-Distill-Qwen-7B"
print("Loading tokenizer from", model_name, flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 为确保 BOS/EOS 是单个 token，直接获取 token_id
bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.convert_tokens_to_ids("<s>")
eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.convert_tokens_to_ids("</s>")

start_token = tokenizer.bos_token if tokenizer.bos_token is not None else "<s>"
end_token = tokenizer.eos_token if tokenizer.eos_token is not None else "</s>"

print("Loading model from", model_name, flush=True)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 过于超过256的QA对，设定对应的long=1
MAX_LEN = 256 

def filter_func(example):
    """
    过滤逻辑：构造完整文本后，计算（不添加特殊标记）token 数量，
    若超过 MAX_LEN 则直接过滤掉该样本。
    """
    q = example["question"]
    a = example["answer"]
    q_with_tokens = start_token + q + end_token
    a_with_tokens = start_token + a + end_token
    full_text = q_with_tokens + "\n" + a_with_tokens
    tokens = tokenizer(full_text, add_special_tokens=True, truncation=False)
    return len(tokens["input_ids"]) <= MAX_LEN

print("Filtering long samples (>256 tokens)...", flush=True)
qa_dataset = qa_dataset.filter(filter_func)

def concat_chunks(examples):
    """
    对每个 batch 样本：
      1. 对每个 QA 对构造完整文本的 token 序列： 
         sample = [bos_id] + question token ids + [eos_id] + [bos_id] + answer token ids + [eos_id]
      2. 将所有样本的 token 序列连续拼接在一起
      3. 当缓冲区 token 数达到或超过 MAX_LEN 时，按精确 MAX_LEN 切分成一个 chunk
      4. 最后不足 MAX_LEN 的 token 丢弃
    返回字典包含 "input_ids", "attention_mask" 和 "labels"（均为切分后的 chunk）
    """
    concatenated_input_ids = []
    concatenated_attention_mask = []
    current_tokens = []
    
    for q, a in zip(examples["question"], examples["answer"]):
        # 分别 tokenize，不添加模型自动特殊标记
        q_ids = tokenizer.encode(q, add_special_tokens=False)
        a_ids = tokenizer.encode(a, add_special_tokens=False)
        # 构造完整的 token 序列，确保 BOS/EOS 插入正确
        sample_ids = [bos_id] + q_ids + [eos_id] + [bos_id] + a_ids + [eos_id]
        current_tokens.extend(sample_ids)
        
        while len(current_tokens) >= MAX_LEN:
            chunk = current_tokens[:MAX_LEN]
            concatenated_input_ids.append(chunk)
            concatenated_attention_mask.append([1] * MAX_LEN)
            current_tokens = current_tokens[MAX_LEN:]
    
    #最后剩下的 tokens 如果不足 MAX_LEN 则直接丢弃
    return {
         "input_ids": concatenated_input_ids,
         "attention_mask": concatenated_attention_mask,
         "labels": [chunk.copy() for chunk in concatenated_input_ids]
    }

print("Starting batch concatenation and tokenization...", flush=True)
tokenized_dataset = qa_dataset.map(
    lambda examples: concat_chunks(examples),
    batched=True,
    remove_columns=qa_dataset.column_names
)
print("Batch concatenation and tokenization completed.", flush=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
train_dataset = tokenized_dataset  # 每个拼接后的 chunk 为一个训练样本

# 自定义 Trainer：使用 BitsAndBytes 的 8-bit Adam 优化器
class CustomTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = bnb.optim.Adam8bit(
                self.model.parameters(),
                lr=self.args.learning_rate,
                betas=(0.9, 0.98),  
                eps=1e-8,
                weight_decay=self.args.weight_decay,
            )
        return self.optimizer

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=20,  
    per_device_train_batch_size=33,
    per_device_eval_batch_size=3,
    gradient_accumulation_steps=8, 
    fp16=True,
    bf16=False,
    auto_find_batch_size=True,  #自动调整 batch size
    deepspeed=args.deepspeed,
    remove_unused_columns=False,
    logging_steps=50,
    eval_strategy="no",
    save_total_limit=3,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay
)

print("Initializing CustomTrainer...", flush=True)
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,  
    data_collator=data_collator
)
print("CustomTrainer initialized. Starting training...", flush=True)

try:
    trainer.train()
except Exception as e:
    print("Training error:", e, flush=True)
    traceback.print_exc()
    sys.exit(1)

print("Training completed.", flush=True)

#保存
final_model_dir = "./ds7b_v1_model"
trainer.save_model(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
print(f"Model and tokenizer saved to {final_model_dir}", flush=True)
