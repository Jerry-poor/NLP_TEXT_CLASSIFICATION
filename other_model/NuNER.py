import os
import pandas as pd
import torch
from transformers import (
    AutoModelForTokenClassification, AutoTokenizer, 
    DataCollatorForTokenClassification, Trainer, 
    TrainingArguments, pipeline
)
from sklearn.model_selection import train_test_split

# 标签映射
def dataset_to_spacy(label):
    mapping = {
        'per': 'PERSON',
        'org': 'ORG',
        'gpe': 'GPE',
        'geo': 'LOC',
        'tim': 'DATE',
        'art': 'WORK_OF_ART',
        'eve': 'EVENT',
        'nat': 'LOC'
    }
    return mapping.get(label, label)

# 用于评估的预处理：每行返回 sentence 与其实体列表（若无实体则为 None）
def process_data_eval(df):
    texts, labels = [], []
    for _, row in df.iterrows():
        texts.append(row['sentence'])
        try:
            entities = eval(row['entities'])
        except Exception:
            entities = []
        if not entities:
            labels.append(None)
        else:
            # 实体格式：[实体文本, (start, end), 类型]
            entity_labels = [[ent[0], [ent[1][0], ent[1][1]], dataset_to_spacy(ent[2])] for ent in entities]
            labels.append(entity_labels)
    return texts, labels

# 定义 BIO 格式标签
label_list = [
    "O",
    "B-PERSON", "I-PERSON",
    "B-ORG", "I-ORG",
    "B-GPE", "I-GPE",
    "B-LOC", "I-LOC",
    "B-DATE", "I-DATE",
    "B-WORK_OF_ART", "I-WORK_OF_ART",
    "B-EVENT", "I-EVENT"
]
label_map = {label: i for i, label in enumerate(label_list)}

# 将字符级实体标注转换为 token 级 BIO 标签
def tokenize_and_align_labels(example, tokenizer, max_length=128):
    text = example['sentence']
    entities = []
    try:
        ents = eval(str(example['entities'])) if example['entities'] is not None else []
        for ent in ents:
            entities.append({
                "text": ent[0],
                "start": ent[1][0],
                "end": ent[1][1] + 1,  # 加1确保token边界匹配
                "label": dataset_to_spacy(ent[2])
            })
    except Exception:
        pass
    tokenized = tokenizer(
        text, padding="max_length", truncation=True, max_length=max_length, return_offsets_mapping=True
    )
    labels_out = []
    for offset in tokenized["offset_mapping"]:
        if offset[0] == offset[1]:
            labels_out.append(-100)
        else:
            token_label = "O"
            for ent in entities:
                if offset[0] >= ent["start"] and offset[1] <= ent["end"]:
                    token_label = "B-" + ent["label"] if offset[0] == ent["start"] else "I-" + ent["label"]
                    break
            labels_out.append(label_map.get(token_label, 0))
    tokenized["labels"] = labels_out
    tokenized.pop("offset_mapping")
    return tokenized

# 聚合多个句子，确保不会出现单个句子部分截断
class AggregatedNERDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        current_text = ""
        current_entities = []
        for _, row in df.iterrows():
            sentence = row['sentence']
            try:
                entities = eval(row['entities'])
            except Exception:
                # 如果该行无法解析实体，则跳过该行
                continue
            add_len = len(sentence) if not current_text else len(sentence) + 1
            if len(current_text) + add_len > max_length:
                self.data.append({
                    'sentence': current_text,
                    'entities': current_entities if current_entities else None
                })
                current_text, current_entities = "", []
            offset = len(current_text) + (1 if current_text else 0)
            current_text = current_text + (" " if current_text else "") + sentence
            for ent in entities:
                new_ent = (ent[0], ent[1][0] + offset, ent[1][1] + offset, ent[2])
                current_entities.append(new_ent)
        if current_text:
            self.data.append({
                'sentence': current_text,
                'entities': current_entities if current_entities else None
            })
    def __len__(self):
         return len(self.data)
    def __getitem__(self, idx):
         sample = self.data[idx]
         return tokenize_and_align_labels(sample, self.tokenizer, self.max_length)

# 简单评估函数：对比 pipeline 的预测与真实实体
def evaluate_model(pipe, texts, labels):
    total, correct = 0, 0
    errors = []
    for text, true_ents in zip(texts, labels):
        preds = pipe(text, aggregation_strategy="simple")
        if true_ents is None:
            continue
        for true_ent in true_ents:
            match = None
            for pred in preds:
                if pred["word"].strip() == true_ent[0].strip():
                    match = pred
                    break
            total += 1
            if match and match["entity_group"] == true_ent[2]:
                correct += 1
            else:
                errors.append({
                    'Sentence': text,
                    'Original_Entity': true_ent,
                    'Predicted_Entity': match if match else {}
                })
    print(f"Overall Label Accuracy: {(correct/total)*100:.2f}%")
    return errors

# 新增：统计有效行和实体数量（只统计能被正确解析的行）
def count_valid_data(df):
    valid_rows = 0
    total_entities = 0
    for _, row in df.iterrows():
        try:
            entities = eval(row['entities'])
            valid_rows += 1
            total_entities += len(entities)
        except Exception:
            continue
    return valid_rows, total_entities

if __name__ == '__main__':
    # 读取 CSV（示例中取前1000行）
    df = pd.read_csv('../dataset/dataset0/abhinavwalia95.csv', encoding='latin1').ffill().head(1000)
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    
    # 在训练时统计有效行数和实体数量（错误行会被跳过）
    train_valid_rows, train_valid_entities = count_valid_data(train_df)
    test_valid_rows, test_valid_entities = count_valid_data(test_df)
    print(f"训练集有效行数: {train_valid_rows}, 实体数量: {train_valid_entities}")
    print(f"验证集有效行数: {test_valid_rows}, 实体数量: {test_valid_entities}")
    
    # 加载模型和 tokenizer，加载初始权重进行微调
    model_name = 'numind/NuNER-v2.0'
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    max_length = 128
    train_dataset = AggregatedNERDataset(train_df, tokenizer, max_length=max_length)
    eval_dataset = AggregatedNERDataset(test_df, tokenizer, max_length=max_length)
    
    data_collator = DataCollatorForTokenClassification(tokenizer)
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        logging_steps=100,
        save_steps=500,
        seed=42,
        disable_tqdm=False,
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
    trainer.save_model("./fine_tuned_model")
    
    pipe = pipeline("token-classification", model=model, tokenizer=tokenizer)
    texts, eval_labels = process_data_eval(test_df)
    errors = evaluate_model(pipe, texts, eval_labels)
    
    # 保存错误预测（可选）
    pd.DataFrame(errors).to_csv("error_results.csv", index=False)
