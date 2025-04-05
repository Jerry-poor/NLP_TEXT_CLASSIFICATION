import pandas as pd
import re

df = pd.read_csv("ner.csv", encoding='ISO-8859-1', on_bad_lines='skip')

# 初始化存储变量
sentences = []
entities = []
current_sentence = []
current_entities = []
current_entity = None
entity_order = 0   # 用于记录当前句子中实体出现的顺序
bad_sentence = False
current = None

# 遍历每一行，主要需要的列为 sentence_idx, word, tag
for index, row in df.iterrows():
    s = row["sentence_idx"]
    w = row["word"]
    t = row["tag"]
    
    # 如果任一必要字段缺失，则标记整句为异常句，后续不处理
    if pd.isna(s) or pd.isna(w) or pd.isna(t):
        bad_sentence = True
        continue

    # 判断是否遇到新句子
    if current is None:
        current = s
    elif s != current:
        # 当句子切换时，如果当前句子不为空且没有错误
        if current_sentence:
            if not bad_sentence:
                # 如果存在正在构造的实体，先将其保存
                if current_entity is not None:
                    current_entities.append([current_entity[0], entity_order, current_entity[1]])
                    entity_order += 1
                sentences.append(" ".join(current_sentence))
                entities.append(current_entities)
        # 重置当前句子相关变量
        current = s
        current_sentence = []
        current_entities = []
        current_entity = None
        entity_order = 0
        bad_sentence = False

    if bad_sentence:
        continue

    # 将当前单词加入句子
    current_sentence.append(w)
    
    # 根据BIO标签处理实体
    if str(t).startswith("B-"):
        # 遇到新的实体开始，如果之前已有未结束的实体，先保存
        if current_entity is not None:
            current_entities.append([current_entity[0], entity_order, current_entity[1]])
            entity_order += 1
        # 初始化新的实体，[文本, 标签]，标签去掉 "B-" 前缀
        current_entity = [w, str(t)[2:]]
    elif str(t).startswith("I-") and current_entity is not None:
        # 续接上一个实体，追加单词
        current_entity[0] += " " + w
    else:
        # 非实体标注（例如"O"）时，如果有正在构造的实体则保存并重置
        if current_entity is not None:
            current_entities.append([current_entity[0], entity_order, current_entity[1]])
            entity_order += 1
            current_entity = None

# 循环结束后，处理最后一句（如果没有错误）
if current_sentence and not bad_sentence:
    if current_entity is not None:
        current_entities.append([current_entity[0], entity_order, current_entity[1]])
    sentences.append(" ".join(current_sentence))
    entities.append(current_entities)

# 输出结果到新的 CSV 文件
output_df = pd.DataFrame({
    "sentence": sentences,
    "entities": [str(entity) for entity in entities]
})
output_df.to_csv("order.csv", index=False)
