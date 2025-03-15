import pandas as pd
import ast

df = pd.read_csv("./dataset1/dataset1_test.csv")

def extract_labels(entities):
    #字符串转换为列表
    if isinstance(entities, str):
        try:
            entities = ast.literal_eval(entities)
        except Exception:
            return []  
    if not isinstance(entities, list):
        return []
    
    # 历每个实体记录，每个记录应为 [entity_name, entity_position, entity_label]
    labels = []
    for item in entities:
        if isinstance(item, list) and len(item) >= 3:
            labels.append(item[2])
    return labels


df['labels'] = df['entities'].apply(extract_labels)

#合并所有行的 labels，去重后打印
all_labels = set(label for labels in df['labels'] for label in labels)
print("Unique Labels:", all_labels)
