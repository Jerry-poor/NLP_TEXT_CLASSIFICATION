import os
import pandas as pd
import torch
from transformers import (
    AutoModelForTokenClassification, AutoTokenizer, 
    DataCollatorForTokenClassification, Trainer, 
    TrainingArguments, pipeline
)
from sklearn.model_selection import train_test_split
import wandb

# ----------------------------
# 使用 wandb 追踪（项目名称可根据需要修改）
# ----------------------------
wandb.login(key="24acbdd88bdf256537ac0b5be8a09ae76abc6604")
wandb.init(project="NuNER_project", name="NuNER0")

# ----------------------------
# 数据预处理：读取 CSV 文件并构造句子
# 该 CSV 文件仅包含两列：word 和 tag。空行（word 为空或仅空白）作为句子分隔符。
# ----------------------------
csv_path = "C:/Users/UIC/Desktop/dataset0_BIO.csv"
df = pd.read_csv(csv_path)

sentences = []
tags = []
current_sentence = []
current_tags = []
for idx, row in df.iterrows():
    word = str(row["word"]).strip() if pd.notna(row["word"]) else ""
    tag = str(row["tag"]).strip() if pd.notna(row["tag"]) else ""
    if word == "":  # 遇到空行，视为句子结束
        if current_sentence:
            sentences.append(current_sentence)
            tags.append(current_tags)
            current_sentence = []
            current_tags = []
    else:
        current_sentence.append(word)
        current_tags.append(tag)
# 如果最后还有未结束的句子，则加入列表
if current_sentence:
    sentences.append(current_sentence)
    tags.append(current_tags)

data_list = [{"tokens": tokens, "tags": tag_list} for tokens, tag_list in zip(sentences, tags)]
print(f"共构造 {len(data_list)} 句子。")

# 划分训练集和验证集
train_data, test_data = train_test_split(data_list, test_size=0.3, random_state=42)

# 构造标签映射（忽略统计数字，只保留标签名称）
all_tags = [tag for tag_list in tags for tag in tag_list]
unique_labels = sorted(set(all_tags))
label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
id_to_label = {idx: label for label, idx in label_to_id.items()}
print("标签映射：", label_to_id)

max_length = 512

# ----------------------------
# 加载 NuNER 预训练模型及 Tokenizer
# ----------------------------
pretrained_model_name = 'numind/NuNER-v2.0'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained(
    pretrained_model_name,
    num_labels=len(unique_labels),
    id2label=id_to_label,
    label2id=label_to_id
)

if torch.cuda.is_available():
    model.to("cuda")
    print("Using GPU for training.")
else:
    print("GPU not available, using CPU.")

# ----------------------------
# 定义自定义 Dataset：利用 tokenizer 的 is_split_into_words 对已分词句子进行编码，并对齐标签
# ----------------------------
class TokenClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, tokenizer, max_length=128):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        example = self.data_list[idx]
        tokenized_inputs = self.tokenizer(
            example["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )
        # 对齐 token 与原始单词的标签：只在每个单词的首个 token 保留标签，其余 token 记为 -100
        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[example["tags"][word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        tokenized_inputs["labels"] = label_ids
        return tokenized_inputs

train_dataset = TokenClassificationDataset(train_data, tokenizer, max_length=max_length)
eval_dataset = TokenClassificationDataset(test_data, tokenizer, max_length=max_length)

# ----------------------------
# 配置 Trainer 及训练参数，开启 wandb 追踪，并将输出目录设置在当前文件夹下的 "NuNER0" 目录
# ----------------------------
data_collator = DataCollatorForTokenClassification(tokenizer)
training_args = TrainingArguments(
    output_dir='./NuNER0',              # 模型保存到当前文件夹下 NuNER0 目录中
    num_train_epochs=50,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    logging_steps=100,
    save_steps=500,
    seed=42,
    disable_tqdm=False,
    fp16=True,                        # 开启混合精度训练以充分利用 GPU
    no_cuda=False,                    # 允许使用 GPU
    run_name="NuNER0",                # 训练名称为 NuNER0
    report_to=["wandb"]               # 启用 wandb 追踪
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./NuNER0")  # 将微调后的模型存储到当前文件夹下的 NuNER0 目录中

# ----------------------------
# 定义仅 token 级别的评估函数（计算精确度、召回率和 F1）
# ----------------------------
def evaluate_token_level(model, eval_dataset):
    model.eval()
    token_correct = 0    # 预测正确的 token 数（仅统计非 "O" 标签）
    token_pred_total = 0 # 模型预测为非 "O" 的 token 数
    token_true_total = 0 # 实际非 "O" 的 token 数

    for example in eval_dataset:
        input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(model.device)
        attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to(model.device)
        labels = example["labels"]  # 序列长度，包含 -100 忽略项
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (1, seq_len, num_labels)
        predictions = torch.argmax(logits, dim=-1).squeeze(0).tolist()  # 长度为 seq_len

        # 对每个 token（忽略 -100）进行统计
        for pred, true in zip(predictions, labels):
            if true == -100:
                continue
            # "O" 标签对应于 label_to_id["O"]
            if true != label_to_id["O"]:
                token_true_total += 1
                if pred == true:
                    token_correct += 1
            if pred != label_to_id["O"]:
                token_pred_total += 1

    precision = token_correct / token_pred_total if token_pred_total > 0 else 0
    recall = token_correct / token_true_total if token_true_total > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print("Token-level Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%".format(
        precision * 100, recall * 100, f1 * 100))
    # 传给 wandb 三种指标
    wandb.log({"token_precision": precision, "token_recall": recall, "token_f1": f1})
    return precision, recall, f1

# ----------------------------
# 在验证集上评估模型（仅 token 级指标）
# ----------------------------
precision, recall, f1 = evaluate_token_level(model, eval_dataset)

