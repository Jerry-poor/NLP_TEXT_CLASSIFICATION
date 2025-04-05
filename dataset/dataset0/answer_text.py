import pandas as pd
import json
import ast

def generate_input(sentence):
    """
    Generate the 'input' field by escaping the sentence using json.dumps.
    This ensures special characters in the sentence are safely embedded.
    """
    return json.dumps(sentence, ensure_ascii=False)

def generate_output(entities_list):
    """
    Convert the entity list into standard JSON format:
      - Input entities_list format: [['Entity1', 'type'], ['Entity2', 'type'], ...]
      - For duplicate entities, 'order' starts from 0 and increments for each subsequent occurrence.
    Returns a JSON string, e.g.:
      {"entities": [{"entity": "Entity1", "order": 0, "label": "type"}, ...]}
    Note: Entities are assumed not to include punctuation so no escaping is needed.
    """
    result_entities = []
    counter = {}
    for ent in entities_list:
        if not isinstance(ent, (list, tuple)) or len(ent) < 2:
            continue
        entity = str(ent[0])
        label = str(ent[1])
        order = counter.get(entity, 0)
        counter[entity] = order + 1
        result_entities.append({"entity": entity, "order": order, "label": label})
    return json.dumps({"entities": result_entities}, ensure_ascii=False)

# -------------------------
# 数据预处理部分：构造句子和实体列表（基于 ner.csv 中的 sentence_idx, word, tag）
# -------------------------
df = pd.read_csv("ner.csv", encoding='ISO-8859-1', on_bad_lines='skip')

sentences = []
entities_data = []
current_sentence = []
current_entities = []
current_entity = None
bad_sentence = False
current = None

for index, row in df.iterrows():
    s = row["sentence_idx"]
    w = row["word"]
    t = row["tag"]
    
    # 若任一关键字段缺失，则标记整个句子为错误并跳过
    if pd.isna(s) or pd.isna(w) or pd.isna(t):
        bad_sentence = True
        continue
    
    # 判断是否遇到新句子
    if current is None:
        current = s
    elif s != current:
        if current_sentence and not bad_sentence:
            if current_entity is not None:
                current_entities.append([current_entity[0], current_entity[1]])
            sentences.append(" ".join(current_sentence))
            entities_data.append(current_entities)
        # 重置变量以开始新句子的处理
        current = s
        current_sentence = []
        current_entities = []
        current_entity = None
        bad_sentence = False
    
    if bad_sentence:
        continue
    
    # 将当前单词加入句子
    current_sentence.append(w)
    
    # 按 BIO 标签规则构造实体
    if str(t).startswith("B-"):
        if current_entity is not None:
            current_entities.append([current_entity[0], current_entity[1]])
        current_entity = [w, str(t)[2:]]
    elif str(t).startswith("I-") and current_entity is not None:
        current_entity[0] += " " + w
    else:
        if current_entity is not None:
            current_entities.append([current_entity[0], current_entity[1]])
            current_entity = None

# 处理最后一句
if current_sentence and not bad_sentence:
    if current_entity is not None:
        current_entities.append([current_entity[0], current_entity[1]])
    sentences.append(" ".join(current_sentence))
    entities_data.append(current_entities)

# -------------------------
# 构造新的 DataFrame，新增 input 和 output 列
# -------------------------
output_df = pd.DataFrame({
    "sentence": sentences,
    "entities": [str(entity) for entity in entities_data]  # 原始实体列表的字符串表示
})

# input 列由句子转义而来
output_df["input"] = output_df["sentence"].apply(generate_input)
# output 列为实体列表的 JSON 改写（不需要额外转义）
output_df["output"] = [generate_output(e_list) for e_list in entities_data]

# 保存预处理后的数据集到 CSV（使用支持中文的 UTF-8 编码）
output_df.to_csv("preprocessed_dataset.csv", index=False, encoding="utf-8-sig")
print("Preprocessing complete. Preprocessed data saved to preprocessed_dataset.csv")
