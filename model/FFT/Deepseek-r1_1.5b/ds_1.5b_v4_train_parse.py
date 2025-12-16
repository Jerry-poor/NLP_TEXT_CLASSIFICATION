import re
import json
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

# ---------------- 1. 载入测试集 Gold 数据 ----------------
df = pd.read_csv("preprocessed_dataset.csv", encoding="utf-8-sig")
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
gold_outputs = train_df['output']

def is_json_complete(json_str):
    """
    简单检测 JSON 字符串是否完整（是否以 '}' 结尾）
    """
    return json_str.strip().endswith("}")

def evaluate_entities(gold_entities, pred_entities):
    """
    采用不去重的方式计算：
      - TP: gold中存在且预测存在且标签匹配的次数（取两者出现次数的较小值）
      - FP: 预测中超过 gold 的部分（包括预测错误的标签）
      - FN: gold中存在但预测缺失或标签错误的次数
    """
    gold_counter = Counter((e['entity'], e['order'], e['label']) for e in gold_entities)
    pred_counter = Counter((e['entity'], e['order'], e['label']) for e in pred_entities)
    tp = sum(min(gold_counter[k], pred_counter.get(k, 0)) for k in gold_counter)
    fp = sum(pred_counter[k] - gold_counter.get(k, 0)
             for k in pred_counter if pred_counter[k] > gold_counter.get(k, 0))
    fn = sum(gold_counter[k] - pred_counter.get(k, 0)
             for k in gold_counter if gold_counter[k] > pred_counter.get(k, 0))
    return tp, fp, fn

# ---------------- 2. 初始化统计变量 ----------------
tp_total = 0
fp_total = 0
fn_total = 0

evaluated_sentence_count = 0  # 有效样本数量（成功处理的样本数）
correct_sentence_count = 0    # 句子级完全匹配数量（基于多重集比较）
total_gold_count = 0          # 数据集中 gold 标注的实体总数（不去重）

# 新增：收集出错样本
error_samples = []

# ---------------- 3. 读取模型输出文件内容 ----------------
with open("ds_1.5b_v4_train_result.txt", "r", encoding="utf-8") as f:
    content = f.read()

# 按分隔符拆分样本
samples = [s.strip() for s in content.split("------------------------") if s.strip()]

# ---------------- 4. 循环处理样本，同时匹配 gold_outputs ----------------
gold_idx = 0

for sample in samples:
    if gold_idx >= len(gold_outputs):
        break

    # 取 gold JSON
    try:
        gold_json = json.loads(gold_outputs.iloc[gold_idx])
    except json.JSONDecodeError:
        gold_idx += 1
        continue

    # 提取模型输出
    pred_match = re.search(r"Model Output:.*?<｜begin▁of▁sentence｜>(\{.*)", sample, re.DOTALL)
    if not pred_match:
        gold_idx += 1
        continue
    pred_str = pred_match.group(1).strip()
    if not is_json_complete(pred_str):
        gold_idx += 1
        continue
    try:
        pred_json = json.loads(pred_str)
    except json.JSONDecodeError:
        gold_idx += 1
        continue

    evaluated_sentence_count += 1

    gold_entities = gold_json.get("entities", [])
    pred_entities = pred_json.get("entities", [])

    # 句子级评估
    gold_counter = Counter((e['entity'], e['order'], e['label']) for e in gold_entities)
    pred_counter = Counter((e['entity'], e['order'], e['label']) for e in pred_entities)
    if gold_counter == pred_counter:
        correct_sentence_count += 1
    else:
        # 记录出错样本
        error_samples.append({
            'sentence': train_df.iloc[gold_idx]['sentence'],
            'true_output': gold_outputs.iloc[gold_idx],
            'model_output': pred_str
        })

    # 实体级评估
    tp, fp, fn = evaluate_entities(gold_entities, pred_entities)
    tp_total += tp
    fp_total += fp
    fn_total += fn

    total_gold_count += len(gold_entities)
    gold_idx += 1

# ---------------- 5. 计算指标 ----------------
def safe_div(a, b):
    return a / b if b else 0

accuracy       = safe_div(tp_total, tp_total + fp_total)
recall_entity  = safe_div(tp_total, total_gold_count)
f1_entity      = safe_div(2 * accuracy * recall_entity, (accuracy + recall_entity))
sentence_prec  = safe_div(correct_sentence_count, evaluated_sentence_count)

# ---------------- 6. 输出评估结果 ----------------
print("Sentence-level evaluation:")
print(f"  Evaluated sentences: {evaluated_sentence_count}")
print(f"  Correct sentences:   {correct_sentence_count}")
print(f"  Sentence-level precision: {sentence_prec:.4f}")

print("\nEntity-level evaluation :")
print(f"  Total gold entities: {total_gold_count}")
print(f"  Predicted correct entities (TP): {tp_total}")
print(f"  Predicted wrong entities (FP): {fp_total}")
print(f"  False negatives (FN): {fn_total}")
print(f"  Accuracy : {accuracy:.4f}")
print(f"  Recall:   {recall_entity:.4f}")
print(f"  F1-score: {f1_entity:.4f}")

# ---------------- 7. 导出出错样本 ----------------
error_df = pd.DataFrame(error_samples)
error_df.to_csv("error_analysis.csv", index=False, encoding="utf-8-sig")
print(f"\n导出出错样本 CSV，共 {len(error_samples)} 条，文件名：error_analysis.csv")
