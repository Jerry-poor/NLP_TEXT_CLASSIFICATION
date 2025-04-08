import re
import json
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

# ---------------- 1. 载入测试集 Gold 数据 ----------------
df = pd.read_csv("preprocessed_dataset.csv", encoding="latin1")
_, test_df = train_test_split(df, test_size=0.3, random_state=42)
# 取前1000行作为评估标准答案（gold），注意这些是原始标注内容
gold_outputs = test_df['output'].iloc[:1000].tolist()

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
    # 将 gold 和 pred 转换为多重集（Counter），以元组 (entity, order, label) 为单位
    gold_counter = Counter((e['entity'], e['order'], e['label']) for e in gold_entities)
    pred_counter = Counter((e['entity'], e['order'], e['label']) for e in pred_entities)
    
    # 计算 TP：对于每个实体，正确预测的次数为 gold 与 pred 出现次数的较小值
    tp = sum(min(gold_counter[k], pred_counter.get(k, 0)) for k in gold_counter)
    
    # FP：预测中多出的部分
    fp = sum(pred_counter[k] - gold_counter.get(k, 0) for k in pred_counter if pred_counter[k] > gold_counter.get(k, 0))
    
    # FN：gold 中没有被预测出来的部分
    fn = sum(gold_counter[k] - pred_counter.get(k, 0) for k in gold_counter if gold_counter[k] > pred_counter.get(k, 0))
    
    return tp, fp, fn

# ---------------- 2. 初始化统计变量 ----------------
tp_total = 0
fp_total = 0
fn_total = 0

evaluated_sentence_count = 0  # 有效样本数量（成功处理的样本数）
correct_sentence_count = 0    # 句子级完全匹配数量（基于多重集比较）
total_gold_count = 0          # 数据集中 gold 标注的实体总数（不去重）

# ---------------- 3. 读取模型输出文件内容 ----------------
with open("combined_eval_results_v4.txt", "r", encoding="latin1") as f:
    content = f.read()

# 按分隔符拆分样本，每个样本以 "------------------------" 为分隔符
samples = [s.strip() for s in content.split("------------------------") if s.strip()]

# ---------------- 4. 循环处理样本，同时匹配 gold_outputs ----------------
gold_idx = 0  # gold_outputs 的索引

for sample in samples:
    # 若 gold_idx 超出范围，则退出
    if gold_idx >= len(gold_outputs):
        break

    # 从 gold_outputs 中获取当前标准答案
    try:
        gold_json = json.loads(gold_outputs[gold_idx])
    except json.JSONDecodeError:
        gold_idx += 1
        continue

    # 提取模型输出部分 JSON（使用自定义标记）
    pred_match = re.search(
        r'Model Output:[\s\S]*?(\{"entities":\s*\[.*?\]\})',
        sample,
        re.DOTALL
    )
    if not pred_match:
        gold_idx += 1  # 无预测输出，跳过并使 gold_idx 前进
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

    # 当前样本有效，对 gold 与预测进行评估
    evaluated_sentence_count += 1

    gold_entities = gold_json.get("entities", [])
    pred_entities = pred_json.get("entities", [])

    # 句子级评估：使用多重集（Counter）比较两个列表的所有元素
    gold_counter = Counter((item['entity'], item['order'], item['label']) for item in gold_entities)
    pred_counter = Counter((item['entity'], item['order'], item['label']) for item in pred_entities)
    if gold_counter == pred_counter:
        correct_sentence_count += 1

    # 实体级评估
    tp, fp, fn = evaluate_entities(gold_entities, pred_entities)
    tp_total += tp
    fp_total += fp
    fn_total += fn

    # 累计 gold 的实体总数（不去重）
    total_gold_count += len(gold_entities)

    # 无论当前样本是否有效，都自增 gold_idx
    gold_idx += 1

# ---------------- 5. 计算指标 ----------------
def safe_div(a, b):
    return a / b if b else 0

# 预测准确度（这里即精确率）：预测正确总数除以模型预测总数
accuracy = safe_div(tp_total, tp_total + fp_total)
# 召回率：预测正确总数除以数据集标注总数（不去重）
recall_entity = safe_div(tp_total, total_gold_count)
f1_entity = safe_div(2 * accuracy * recall_entity, (accuracy + recall_entity))

sentence_precision = safe_div(correct_sentence_count, evaluated_sentence_count)

# ---------------- 6. 输出评估结果 ----------------
print("Sentence-level evaluation:")
print("  Evaluated sentences: {}".format(evaluated_sentence_count))
print("  Correct sentences: {}".format(correct_sentence_count))
print("  Sentence-level precision: {:.4f}".format(sentence_precision))

print("\nEntity-level evaluation :")
print("  Total gold entities: {}".format(total_gold_count))
print("  Predicted correct entities (TP): {}".format(tp_total))
print("  Predicted wrong entities (FP): {}".format(fp_total))
print("  False negatives (FN): {}".format(fn_total))
print("  Accuracy : {:.4f}".format(accuracy))
print("  Recall: {:.4f}".format(recall_entity))
print("  F1-score: {:.4f}".format(f1_entity))
