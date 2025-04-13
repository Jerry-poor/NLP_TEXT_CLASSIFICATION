import os
import re
import json
import ast
import pandas as pd

base_path = 'chat'
data = []  # 用于保存所有批次的预测结果，格式：(sentence_index, predicted_entities)

# 定义用于提取 response 字段的正则（作为 JSON 解析失败的回退）
resp_pattern = re.compile(r'"response":\s*"(.*?)"\s*}', re.DOTALL)
# 定义用于提取实体的正则，匹配形如 [ "实体文本", (pos, pos), "标签" ]
entity_pattern = re.compile(r"\[\s*(['\"])(.*?)\1\s*,\s*\(.*?\)\s*,\s*(['\"])(.*?)\3\s*\]")

# 遍历 batch_id 从 1 到 8
for batch_id in range(1, 9):
    file_path = os.path.join(base_path, f'test_batch{batch_id}.txt')
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

        # 尝试先通过 JSON 解析获取 response 字段
        try:
            content_dict = json.loads(content)
            response_text = content_dict.get('response', '')
        except Exception as e:
            print(f"JSON parse failed in file {file_path} with error: {e}")
            # JSON 解析失败，回退使用正则抽取 response 字段
            m = resp_pattern.search(content)
            if m:
                response_text = m.group(1).replace("\\n", "\n")
                print(f"Using regex fallback for file {file_path}")
            else:
                print(f"Neither JSON nor regex extraction succeeded for file {file_path}")
                continue

        # 将 response_text 按换行符拆分，每行预期对应一句话
        lines = response_text.split("\n")
        batch_data = []  # 保存当前批次中每句话的预测结果（列表形式）
        for line in lines:
            line = line.strip()
            # 如果为空或仅为 "[]"（空列表），则返回空列表
            if not line or line == "[]":
                batch_data.append([])
                continue
            # 预处理：将转义的引号转换成正常引号
            line_clean = line.replace('\\"', '"').replace("\\'", "'")
            # 用正则抽取所有实体信息，仅提取实体文本和标签（忽略位置）
            matches = entity_pattern.findall(line_clean)
            if matches:
                entities = []
                for match in matches:
                    # match: (quote, entity_text, quote, label)
                    entity_text = match[1].strip()
                    label = match[3].strip()
                    entities.append([entity_text, label])
                batch_data.append(entities)
            else:
                batch_data.append([])

        # 计算当前批次的起始句子索引（每批固定 50 行，首句号为 1002）
        start_idx = 1001 + (batch_id - 1) * 50
        # 为当前批次每句话构造 (sentence_index, prediction) 元组
        data.extend([(start_idx + i, row) for i, row in enumerate(batch_data)])

# 若没有解析到任何数据，则报错
if not data:
    raise RuntimeError("No valid data parsed from chat/*.txt files.")

# 用预测结果构造 DataFrame，索引为句子编号，列 "output" 为预测的实体列表
idx, rows = zip(*data)
df_pred = pd.DataFrame({'output': rows}, index=pd.Index(idx, name='sentence'))

# 将预测结果写入结构化文本文件，每行格式为 [[entity_name, label], ...]
output_file = "structured_predictions.txt"
with open(output_file, 'w', encoding='utf-8') as fout:
    for i, pred in sorted(data, key=lambda x: x[0]):
        fout.write(str(pred) + "\n")
print(f"结构化预测结果已写入 {output_file}")

# ----------------------------- 读取标注文件 -----------------------------
try:
    gold_df = pd.read_csv('entity_label.csv', index_col=0)
except FileNotFoundError:
    raise FileNotFoundError('entity_label.csv not found.')

# 清洗 gold_df 索引：转为数值、移除非数值行，然后转换为整数
gold_df.index = pd.to_numeric(gold_df.index, errors='coerce')
gold_df = gold_df[gold_df.index.notnull()]
gold_df.index = gold_df.index.astype(int)

# 自动识别 gold_df 中实体列（如果存在 "entities" 列则采用，否则取第一列）
if 'entities' in gold_df.columns:
    ent_col = 'entities'
else:
    ent_col = gold_df.columns[0]

# 对预测结果进行对齐：保留索引同时出现在 gold_df 中的句子
df_pred = df_pred[df_pred.index.isin(gold_df.index)]
df_pred = df_pred.sort_index()
gold_df = gold_df.loc[df_pred.index]  # 对 gold 标注也进行排序，使其与预测严格对应

import ast

# ----------------------------- 指标计算部分 -----------------------------

true_total = 0  # Gold 中所有实体数量
pred_total = 0  # 预测中所有实体数量
correct_ents = 0  # 正确匹配的实体数量
correct_sents = 0  # 句子级准确数量

for i in df_pred.index:
    # 尝试解析 gold 数据（预期为字符串形式的 Python 列表）
    try:
        gold_entities = ast.literal_eval(gold_df.loc[i, ent_col])
    except Exception as e:
        print(f"Error parsing gold data for sentence {i}: {e}")
        gold_entities = []
    # 构建 gold 字典：实体名称为 key，label 为 value
    gold_dict = {}
    for item in gold_entities:
        if isinstance(item, list) and len(item) == 2:
            key = item[0].strip()
            value = item[1].strip()
            gold_dict[key] = value

    # 获取预测结果（已为列表形式）
    pred_entities = df_pred.at[i, 'output']
    pred_dict = {}
    for item in pred_entities:
        if isinstance(item, list) and len(item) == 2:
            key = item[0].strip()
            value = item[1].strip()
            pred_dict[key] = value

    true_total += len(gold_dict)
    pred_total += len(pred_dict)
    
    # 对每个预测实体，若在 gold 中存在且 label 匹配，则计为正确
    correct_count = 0
    for entity, pred_label in pred_dict.items():
        if entity in gold_dict and gold_dict[entity] == pred_label:
            correct_count += 1
    correct_ents += correct_count

    # 句子级：如果预测字典中所有实体均在 gold 中且完全匹配（且预测数量大于 0），认为句子正确
    if pred_dict and all(entity in gold_dict and gold_dict[entity] == pred_dict[entity] for entity in pred_dict):
        correct_sents += 1

precision = correct_ents / pred_total if pred_total else 0.0
recall    = correct_ents / true_total if true_total else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
sent_acc  = correct_sents / len(df_pred) if len(df_pred) else 0.0

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Sentence-level Accuracy: {sent_acc:.4f}")
