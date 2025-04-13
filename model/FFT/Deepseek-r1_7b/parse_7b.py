import re
import json
import pandas as pd
from collections import Counter

def is_json_complete(json_str):
    """
    简单判断 JSON 字符串是否完整（以 '}' 结束）。
    """
    return json_str.strip().endswith("}")

def evaluate_entities(gold_entities, pred_entities):
    """
    计算实体级评估指标：
      - TP: 对于每个 (entity, order, label) 键，取金标准与预测中较小的数量；
      - FP: 预测中多出的部分；
      - FN: 金标准中漏掉的部分。
    """
    gold_counter = Counter((e['entity'], e['order'], e['label']) for e in gold_entities)
    pred_counter = Counter((e['entity'], e['order'], e['label']) for e in pred_entities)
    tp = sum(min(gold_counter[k], pred_counter.get(k, 0)) for k in gold_counter)
    fp = sum(pred_counter[k] - gold_counter.get(k, 0) for k in pred_counter if pred_counter[k] > gold_counter.get(k, 0))
    fn = sum(gold_counter[k] - pred_counter.get(k, 0) for k in gold_counter if gold_counter[k] > pred_counter.get(k, 0))
    return tp, fp, fn


df = pd.read_csv("ds_7b_output.csv", encoding="utf-8")
df = df.iloc[:1000]


tp_total = 0
fp_total = 0
fn_total = 0

gold_token_total_global = 0
pred_token_total_global = 0
correct_token_total_global = 0

evaluated_sentence_count = 0  # 有效评估样本数
correct_sentence_count = 0    # 完全匹配（基于 multiset 比较）的句子数
total_gold_count = 0          # 金标准实体总数

# 使用宽松匹配：只要标记内包含 "begin" 和 "sentence"，以及 "end" 和 "sentence" 的关键词，就捕获中间内容
pattern = re.compile(r"<.*?begin.*?sentence.*?>(.*?)<.*?end.*?sentence.*?>", re.DOTALL)

# ---------------- 4. 处理每一行样本 ----------------
for idx, row in df.iterrows():
    try:
        gold_json = json.loads(row["output"])
    except Exception:
        continue

    # 从 model_output 中提取文本
    model_str = row["model_output"]
    
    # 使用宽松正则表达式匹配所有标记之间的内容
    matches = pattern.findall(model_str)
    
    if len(matches) < 2:
        # 没有匹配到至少两组内容，则跳过该行
        continue

    # 取第二组匹配内容作为预测结果（实体 JSON 部分）
    pred_str = matches[1].strip()

    if not is_json_complete(pred_str):
        continue

    try:
        pred_json = json.loads(pred_str)
    except Exception:
        continue

    evaluated_sentence_count += 1

    # 如果需要直接打印抓取到的 entities 内容（仅打印前五个），可以在此处打印
    if evaluated_sentence_count <= 5:
        print(f"Extracted entities JSON for sample {evaluated_sentence_count}:")
        print(pred_str)
        print("=" * 50)

    # 提取实体列表，若 JSON 为 {"entities": []} 则返回空列表
    gold_entities = gold_json.get("entities", [])
    pred_entities = pred_json.get("entities", [])

    # ---------------- 句子级评估 ----------------
    gold_counter = Counter((item['entity'], item['order'], item['label']) for item in gold_entities)
    pred_counter = Counter((item['entity'], item['order'], item['label']) for item in pred_entities)
    if gold_counter == pred_counter:
        correct_sentence_count += 1

    # ---------------- 实体级评估 ----------------
    tp, fp, fn = evaluate_entities(gold_entities, pred_entities)
    tp_total += tp
    fp_total += fp
    fn_total += fn
    total_gold_count += len(gold_entities)

    # ---------------- Token级评估 ----------------
    # 这里假定用空格分割计算 token 数
    gold_token_total_sample = sum(len(e['entity'].split()) for e in gold_entities)
    pred_token_total_sample = sum(len(e['entity'].split()) for e in pred_entities)
    
    gold_counter_tokens = Counter((e['entity'], e['order'], e['label']) for e in gold_entities)
    pred_counter_tokens = Counter((e['entity'], e['order'], e['label']) for e in pred_entities)
    gold_token_length = { (e['entity'], e['order'], e['label']): len(e['entity'].split())
                          for e in gold_entities }
    
    correct_token_sample = sum(
        min(gold_counter_tokens[k], pred_counter_tokens.get(k, 0)) * gold_token_length[k]
        for k in gold_counter_tokens
    )

    gold_token_total_global += gold_token_total_sample
    pred_token_total_global += pred_token_total_sample
    correct_token_total_global += correct_token_sample

# ---------------- 5. 计算各项指标 ----------------
def safe_div(a, b):
    return a / b if b else 0

accuracy_entity = safe_div(tp_total, tp_total + fp_total)
recall_entity = safe_div(tp_total, total_gold_count)
f1_entity = safe_div(2 * accuracy_entity * recall_entity, (accuracy_entity + recall_entity))
sentence_precision = safe_div(correct_sentence_count, evaluated_sentence_count)
token_precision = safe_div(correct_token_total_global, pred_token_total_global)
token_recall = safe_div(correct_token_total_global, gold_token_total_global)
f1_token = safe_div(2 * token_precision * token_recall, (token_precision + token_recall))

# ---------------- 6. 输出评估结果 ----------------
print("Sentence-level evaluation:")
print("  Evaluated sentences: {}".format(evaluated_sentence_count))
print("  Correct sentences: {}".format(correct_sentence_count))
print("  Sentence-level precision: {:.4f}".format(sentence_precision))
print("\nEntity-level evaluation:")
print("  Total gold entities: {}".format(total_gold_count))
print("  Predicted correct entities (TP): {}".format(tp_total))
print("  Predicted wrong entities (FP): {}".format(fp_total))
print("  False negatives (FN): {}".format(fn_total))
print("  Accuracy: {:.4f}".format(accuracy_entity))
print("  Recall: {:.4f}".format(recall_entity))
print("  F1-score: {:.4f}".format(f1_entity))
print("\nToken-level evaluation:")
print("  Total gold tokens: {}".format(gold_token_total_global))
print("  Predicted tokens: {}".format(pred_token_total_global))
print("  Correct tokens: {}".format(correct_token_total_global))
print("  Precision: {:.4f}".format(token_precision))
print("  Recall: {:.4f}".format(token_recall))
print("  F1-score: {:.4f}".format(f1_token))
