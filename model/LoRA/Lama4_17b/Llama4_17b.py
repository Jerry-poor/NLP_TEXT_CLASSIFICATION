#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import pandas as pd
import torch
import random
import wandb
import numpy as np
import traceback
import re
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, set_seed, DataCollatorWithPadding, BitsAndBytesConfig
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, LlamaForCausalLM
from datasets import Dataset
from sklearn.model_selection import train_test_split
import argparse
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from bitsandbytes.nn import Linear4bit
from Llama4_17b_MOE import get_moe_model
parser = argparse.ArgumentParser(description='Fine-tune a model using Deepspeed')
parser.add_argument('--deepspeed', type=str, required=True, help='Path to deepspeed configuration file')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
parser.add_argument('--local_rank', type=int, default=-1, help='Local rank passed from distributed launcher')
parser.add_argument('--config', type=str, required=True,help='Path to your axolotl_config.yaml file')
args = parser.parse_args()
#为了保障可以复现，本次实验将采用随机种子全部设置为42的策略
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
set_seed(SEED)



if args.local_rank == 0:
    api_key = os.environ.get("WANDB_API_KEY")
    print(f"WANDB_API_KEY in env = {api_key}", flush=True)
    if api_key:
        wandb.login(key=api_key, relogin=False)
        wandb.init(project="Llama_lora_finetuning", mode="online")
    else:
        print("⚠️ WANDB_API_KEY is not set; disabling wandb", flush=True)
        wandb.init(project="Llama_lora_finetuning", mode="disabled")

fixed_prompt = (
    "You are a natural language processing assistant. Please extract all named entities from the input text "
    "and return the results in JSON format. If the same entity appears multiple times, assign an increasing order "
    "starting from 0. "
)


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


model_name = "Llama-4-Scout-17B-16E-Instruct"
print("Loading tokenizer from", model_name, flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
START_TOKEN = "<|begin_of_text|>"
END_TOKEN = "<|eot|>"
PAD_TOKEN = "<|finetune_right_pad|>"
tokenizer.add_special_tokens({
        'bos_token': START_TOKEN,
        'eos_token': END_TOKEN,
        'pad_token': PAD_TOKEN
    })
bos_id = tokenizer.convert_tokens_to_ids(START_TOKEN)
eos_id = tokenizer.convert_tokens_to_ids(END_TOKEN)
pad_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)

#清理多模态token，词表不覆盖
vision_keywords = ["image", "vision", "patch", "multi_modal"]
all_tokens = list(tokenizer.get_vocab().keys())
tokens_to_mask = [tok for tok in all_tokens
                  if any(k in tok.lower() for k in vision_keywords)]
token_ids_to_mask = tokenizer.convert_tokens_to_ids(tokens_to_mask)
#清理权重json，这个由于我有备份所以直接覆盖文件
index_file_path = os.path.join(model_name, "model.safetensors.index.json")
with open(index_file_path, "r") as f:
    content = f.read()
    if not content.strip():
        raise ValueError(f"The file is null!：{index_file_path}")
    index_data = json.loads(content)


original_map = index_data.get("weight_map", {})
pruned_map = {
    k: v for k, v in original_map.items()
    if not re.search(r"(vision_model|multi_modal)", k)
}
num_pruned = len(original_map) - len(pruned_map)

index_data["weight_map"] = pruned_map
with open(index_file_path, "w") as f:
    json.dump(index_data, f, indent=2)


#加载剪枝后模型
config_path = os.path.join(model_name, "config.json")
with open(config_path, "r") as f:
    raw_cfg = json.load(f)
text_cfg = raw_cfg.get("text_config", {})
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
#这个有个严重问题，可能是Llama 4太新了，不能解析正确部分权重，所以得手动配置
config.bos_token_id            = bos_id
config.eos_token_id            = eos_id
config.pad_token_id            = pad_id
config.vocab_size            = text_cfg.get("vocab_size",            config.vocab_size)
config.hidden_size           = text_cfg.get("hidden_size",           config.hidden_size)
config.num_attention_heads   = text_cfg.get("num_attention_heads",   config.num_attention_heads)
config.num_hidden_layers     = text_cfg.get("num_hidden_layers",     config.num_hidden_layers)
config.intermediate_size     = text_cfg.get("intermediate_size",     config.intermediate_size)
config.rms_norm_eps          = text_cfg.get("rms_norm_eps",          getattr(config, "rms_norm_eps", None))
config.num_key_value_heads   = text_cfg.get("num_key_value_heads",   getattr(config, "num_key_value_heads", None))
config.attention_dropout     = text_cfg.get("attention_dropout",     getattr(config, "attention_dropout", None))
config.attention_bias        = text_cfg.get("attention_bias",        getattr(config, "attention_bias", None))
config.rope_theta            = text_cfg.get("rope_theta",            getattr(config, "rope_theta", None))
config.initializer_range     = text_cfg.get("initializer_range",     getattr(config, "initializer_range", None))
# MoE 下前馈层没有 bias，可以默认为 False
config.mlp_bias              = text_cfg.get("mlp_bias", False)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 启用 4-bit 量化 :contentReference[oaicite:0]{index=0}
    bnb_4bit_quant_type="nf4",               # 推荐的 NF4 量化类型 :contentReference[oaicite:1]{index=1}
    bnb_4bit_compute_dtype=torch.bfloat16,   # 计算时使用 bfloat16
    bnb_4bit_linear=True,
    bnb_4bit_use_double_quant=True          # 是否使用 double quant
)
# 使用 LlamaForCausalLM 来加载模型，Transformers 内部会根据修改过的 index 文件加载权重
model = LlamaForCausalLM.from_pretrained(model_name, 
                                         config=config,
                                         quantization_config=quant_config,       # 传入 BitsAndBytesConfig :contentReference[oaicite:2]{index=2}
                                         trust_remote_code=True,                 # 允许加载模型仓库中自定义的 llama4 代码 :contentReference[oaicite:3]{index=3}
                                         device_map="auto")
print(model)
for module in model.modules():
    if isinstance(module, Linear4bit):
        # bitsandbytes v0.39+ 里，这两个字段后注入会报错
        module.weight.compress_statistics = None
        module.weight.quant_type = getattr(module, "quant_type", None)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # 针对注意力的 q/v 模块进行微调
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print("Model wrapped for QLoRA training.")
#过滤长样本并批量拼接

MAX_LEN = 256  # 设定每个 chunk 的 token 数

def filter_func(example):
    """
    过滤逻辑：构造完整文本后，计算（不添加特殊标记）token 数量，
    若超过 MAX_LEN 则直接过滤掉该样本。
    """
    return len(tokenizer.encode(ex['question'] + '\n' + ex['answer'], add_special_tokens=False)) <= MAX_LEN
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
    
    # 最后剩下的 tokens 如果不足 MAX_LEN 则直接丢弃
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

print("Starting QLoRA training using Axolotl...", flush=True)

# 加载 Axolotl YAML 配置
with open(args.config, "r") as f:
    axo_cfg = yaml.safe_load(f)

axo_cfg["seed"] = SEED
axo_cfg["load_in_4bit"] = True
axo_cfg["bnb_4bit_quant_type"] = "nf4"
axo_cfg["bnb_4bit_compute_dtype"] = "bfloat16"
axo_cfg["bnb_4bit_use_double_quant"] = True
axo_cfg["deepspeed"] = args.deepspeed

# 启动 QLoRA 微调
trainer = LoraTrainer(
    config=axo_cfg,
    model=model,
    tokenizer=tokenizer,
    dataset=train_dataset,
)
trainer.train()
print("Training completed using Axolotl.", flush=True)

# 保存模型
final_model_dir = "./Llama_v1_model"
trainer.save_model(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
print(f"Model and tokenizer saved to {final_model_dir}", flush=True)
