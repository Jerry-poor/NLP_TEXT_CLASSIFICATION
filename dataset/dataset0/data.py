import pandas as pd
import re
df = pd.read_csv("ner.csv", encoding='ISO-8859-1', on_bad_lines='skip')
#这一版的处理代码考虑对位置进行处理，以及使用新的位置标签。但结构保持[entity_name, entity_poistion, entity_label]
sentences = []
entities = []
current_sentence = []
current_entities = []
current_entity = None
start_pos = None
bad_sentence = False
current = None
poistion = None
prev_rel_pos = 0
#主要的需要的列为sentence,word,tag
for index, row in df.iterrows():
    s = row["sentence_idx"]
    w = row["word"]
    t = row["tag"]
    if pd.isna(s) or pd.isna(w) or pd.isna(t):
        bad_sentence = True
        continue
    if current is None:
        current = s
        poistion = index
    elif s != current:
        if current_sentence:
            if not bad_sentence:
                if current_entity is not None:
                    current_entities.append([current_entity[0], (start_pos, prev_rel_pos), current_entity[1]])
                sentences.append(" ".join(current_sentence))
                entities.append(current_entities)
        current = s
        poistion = index
        current_sentence = []
        current_entities = []
        current_entity = None
        bad_sentence = False
    if bad_sentence:
        continue
    rel_pos = index - poistion
    current_sentence.append(w)
    if str(t).startswith("B-"):
        if current_entity:
            current_entities.append([current_entity[0], (start_pos, prev_rel_pos), current_entity[1]])
        current_entity = [w, str(t)[2:]]
        start_pos = rel_pos
    elif str(t).startswith("I-") and current_entity:
        current_entity[0] += " " + w
    else:
        if current_entity:
            current_entities.append([current_entity[0], (start_pos, rel_pos - 1), current_entity[1]])
            current_entity = None
            start_pos = None
    prev_rel_pos = rel_pos

if current_sentence and not bad_sentence:
    if current_entity:
        current_entities.append([current_entity[0], (start_pos, prev_rel_pos), current_entity[1]])
    sentences.append(" ".join(current_sentence))
    entities.append(current_entities)

output_df = pd.DataFrame({
    "sentence": sentences,
    "entities": [str(entity) for entity in entities]
})
output_df.to_csv("dataset0.csv", index=False)
