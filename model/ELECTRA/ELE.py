import os 
import random
import numpy as np
import torch
import pandas as pd
import wandb

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer, 
    DataCollatorForTokenClassification,
    Trainer, 
    TrainingArguments, 
    set_seed,
)
from sklearn.model_selection import train_test_split

# ─── 0. 固定随机种子 & cuDNN 确定性 ─────────────────────────
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

# ─── 1. WandB 追踪 ─────────────────────────────────────────
wandb.login(key="24acbdd88bdf256537ac0b5be8a09ae76abc6604")
wandb.init(project="Electra_NER_Project", name="Electra_IOB")

# ─── 2. 设备选择 ───────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ─── 3. 读取 IOB 格式 CSV ─────────────────────────────────
csv_path = r"C:/Users/UIC/Desktop/dataset0_BIO.csv"
df = pd.read_csv(csv_path)

sentences, tags = [], []
cur_tokens, cur_tags = [], []
for _, row in df.iterrows():
    w = str(row["word"]).strip() if pd.notna(row["word"]) else ""
    t = str(row["tag"]).strip()  if pd.notna(row["tag"])  else ""
    if w == "":
        if cur_tokens:
            sentences.append(cur_tokens)
            tags.append(cur_tags)
            cur_tokens, cur_tags = [], []
    else:
        cur_tokens.append(w)
        cur_tags.append(t)
if cur_tokens:
    sentences.append(cur_tokens)
    tags.append(cur_tags)

data = [{"tokens": s, "tags": tg} for s, tg in zip(sentences, tags)]
print(f"Loaded {len(data)} sentences.")

train_data, eval_data = train_test_split(data, test_size=0.3, random_state=SEED)

# ─── 4. 构造标签映射 ─────────────────────────────────────────
all_tags      = sum(tags, [])
unique_labels = sorted(set(all_tags))
label2id      = {lbl: i for i, lbl in enumerate(unique_labels)}
id2label      = {i: lbl for lbl, i in label2id.items()}
num_labels    = len(unique_labels)
print("Label mapping:", label2id, "| num_labels =", num_labels)

# ─── 5. 从本地目录加载 ELECTRA 模型 & Tokenizer ────────────────────────
LOCAL_MODEL_DIR = r"C:/Users/UIC/Desktop/ELECTRA/electra_ner_local"
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, add_prefix_space=True)

model = AutoModelForTokenClassification.from_pretrained(
    LOCAL_MODEL_DIR,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True   # ← 忽略旧分类头与新标签数不一致
).to(device)

# ─── 6. 自定义 Dataset ───────────────────────────────────────
class TokenClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, tokenizer, max_length=128):
        self.data       = data_list
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        enc = self.tokenizer(
            example["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )
        word_ids = enc.word_ids()
        prev_word = None
        label_ids = []
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            elif wid != prev_word:
                label_ids.append(label2id[ example["tags"][wid] ])
            else:
                label_ids.append(-100)
            prev_word = wid
        enc["labels"] = label_ids
        return {k: torch.tensor(v) for k, v in enc.items()}

train_ds = TokenClassificationDataset(train_data, tokenizer, max_length=128)
eval_ds  = TokenClassificationDataset(eval_data,  tokenizer, max_length=128)

# ─── 7. Trainer & TrainingArguments ─────────────────────────
data_collator = DataCollatorForTokenClassification(tokenizer)
training_args = TrainingArguments(
    output_dir="./electra_iob",
    num_train_epochs=200,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    logging_steps=100,
    save_steps=500,
    seed=SEED,
    fp16=True,
    no_cuda=False,
    run_name="Electra_IOB_Finetune",
    report_to=["wandb"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# ─── 8. 开始训练 ─────────────────────────────────────────────
trainer.train()
trainer.save_model("./electra_iob")

# ─── 9. 评估示例（Token-level Precision/Recall/F1） ────────────
def evaluate_token_level(model, dataset):
    model.eval()
    tp = fp = fn = 0
    for batch in dataset:
        input_ids = batch["input_ids"].unsqueeze(0).to(device)
        mask      = batch["attention_mask"].unsqueeze(0).to(device)
        labels    = batch["labels"]
        with torch.no_grad():
            logits = model(input_ids, attention_mask=mask).logits
        preds = logits.argmax(-1).squeeze().cpu().tolist()
        for p, t in zip(preds, labels):
            if t == -100:
                continue
            if p == t and t != label2id["O"]:
                tp += 1
            if p != t and p != label2id["O"]:
                fp += 1
            if p != t and t != label2id["O"]:
                fn += 1
    precision = tp / (tp + fp) if tp+fp>0 else 0
    recall    = tp / (tp + fn) if tp+fn>0 else 0
    f1        = 2*precision*recall/(precision+recall) if precision+recall>0 else 0
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    wandb.log({"precision": precision, "recall": recall, "f1": f1})

evaluate_token_level(model, eval_ds)
