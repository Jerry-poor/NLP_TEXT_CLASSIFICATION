import spacy
import os
import pandas as pd
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import minibatch, compounding
from sklearn.model_selection import train_test_split
from eval import evaluate_model

train_df = pd.read_csv("./dataset/dataset1/dataset1_train.csv", encoding='latin1').ffill()
test_df = pd.read_csv("./dataset/dataset1/dataset1_test.csv", encoding='latin1').ffill()
path = './dataset/dataset1/'
model_name = "fine_tuned_model"
nlp = spacy.load("en_core_web_sm")
#建立一个映射，将原始数据集转换成spacy标签，不在表中的说明与spacy标签无法建立映射，后续当none处理
def dataset_to_spacy(label):
    mapping = {
        'PER': 'PERSON',
        'ORG': 'ORG',
        'LOC': 'LOC',
    }
    return mapping.get(label, label)
#由于未知原因，可能是数据集或者spacy的原因，会出现一个实体被多次引用的情况，对于重复span的情况，仅保留第一次的实体
def filter_overlapping_spans_keep_first(spans):
    kept = []
    for span in spans:
        overlap = False
        for kept_span in kept:
            if not (span.start_char >= kept_span.end_char or span.end_char <= kept_span.start_char):
                overlap = True
                break
        if not overlap:
            kept.append(span)
    return kept
def create_doc(nlp, text, entities):
    doc = nlp(text)
    ents = []
    try:
        entity_list = eval(entities)
    except Exception as e:
        entity_list = []
    for entity in entity_list:
        annotated_text = entity[0]  
        start = entity[1][0]
        end = entity[1][1] + 1 
        label = dataset_to_spacy(entity[2])
        # 优先尝试使用 alignment_mode="expand"
        span = doc.char_span(start, end, label=label, alignment_mode="expand")
        if span is None:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            idx = doc.text.find(annotated_text)
            if idx != -1:
                span = doc.char_span(idx, idx + len(annotated_text), label=label, alignment_mode="contract")
        if span is not None:
            ents.append(span)
        else:
            extracted = doc.text[start:end] if start < len(doc.text) and end <= len(doc.text) else "N/A"
    doc.ents = filter_overlapping_spans_keep_first(ents)
    return doc


train_doc_bin = DocBin()
for index, row in train_df.iterrows():
    doc = create_doc(nlp, row['sentence'], row['entities'])
    train_doc_bin.add(doc)
train_doc_bin.to_disk(os.path.join(path, 'train.spacy'))

dev_doc_bin = DocBin()
for index, row in test_df.iterrows():
    doc = create_doc(nlp, row['sentence'], row['entities'])
    dev_doc_bin.add(doc)
dev_doc_bin.to_disk(os.path.join(path, 'dev.spacy'))

def process_data(df):
    texts = []
    labels = []
    for index, row in df.iterrows():
        text = row['sentence']
        texts.append(text)
        entities = row['entities']
        entity_labels = []
        for entity in eval(entities):
            entity_labels.append([entity[0], entity[1], dataset_to_spacy(entity[2])])
        labels.append(entity_labels)
    return texts, labels
train_examples = []
for index, row in train_df.iterrows():
    doc = create_doc(nlp, row['sentence'], row['entities'])
    gold_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
    example = Example.from_dict(doc, {"entities": gold_entities})
    train_examples.append(example)
optimizer = nlp.resume_training()
n_iter = 50
for epoch in range(n_iter):
    losses = {}
    batches = minibatch(train_examples, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        nlp.update(batch, drop=0.5, sgd=optimizer, losses=losses)
    print(f"Epoch {epoch+1}/{n_iter}, Losses: {losses}")

nlp.to_disk(os.path.join(path, model_name))
texts, labels = process_data(test_df)
evaluate_model(nlp, texts, labels, path, model_name)
