import pandas as pd
import re

# 读取数据文件
df = pd.read_csv("ner.csv", encoding='ISO-8859-1', on_bad_lines='skip')

# 初始化变量，用于存储句子和对应的实体
sentences = []
entities = []
current_sentence = []
current_entities = []
current_entity = None
bad_sentence = False
current = None

# 遍历 DataFrame 中的每一行，主要关注列：sentence_idx, word, tag
for index, row in df.iterrows():
    s = row["sentence_idx"]
    w = row["word"]
    t = row["tag"]
    
    # 如果任一关键字段缺失，则将整句标记为错误，跳过处理
    if pd.isna(s) or pd.isna(w) or pd.isna(t):
        bad_sentence = True
        continue
    
    # 判断是否遇到新句子
    if current is None:
        current = s
    elif s != current:
        # 当句子切换时，将上一句的结果保存（如果没有标记为错误）
        if current_sentence:
            if not bad_sentence:
                # 如果有正在构造的实体，先保存
                if current_entity is not None:
                    current_entities.append([current_entity[0], current_entity[1]])
                sentences.append(" ".join(current_sentence))
                entities.append(current_entities)
        # 重置句子相关变量，开始处理新句子
        current = s
        current_sentence = []
        current_entities = []
        current_entity = None
        bad_sentence = False

    # 如果句子被标记为错误，则跳过当前行
    if bad_sentence:
        continue

    # 将当前单词加入句子
    current_sentence.append(w)
    
    # 根据 BIO 标签处理实体
    if str(t).startswith("B-"):
        # 如果之前存在未结束的实体，先保存该实体
        if current_entity is not None:
            current_entities.append([current_entity[0], current_entity[1]])
        # 新实体开始，记录实体文本和标签（去除 "B-" 前缀）
        current_entity = [w, str(t)[2:]]
    elif str(t).startswith("I-") and current_entity is not None:
        # 当前单词属于正在构建的实体，将其追加到实体文本中
        current_entity[0] += " " + w
    else:
        # 遇到“O”标签（或非 B-、I- 标签），如果有未结束的实体，则保存该实体，并重置 current_entity
        if current_entity is not None:
            current_entities.append([current_entity[0], current_entity[1]])
            current_entity = None

# 处理最后一句
if current_sentence and not bad_sentence:
    if current_entity is not None:
        current_entities.append([current_entity[0], current_entity[1]])
    sentences.append(" ".join(current_sentence))
    entities.append(current_entities)

# 将处理结果写入新的 CSV 文件
output_df = pd.DataFrame({
    "sentence": sentences,
    "entities": [str(entity) for entity in entities]
})
output_df.to_csv("entity_label.csv", index=False)
