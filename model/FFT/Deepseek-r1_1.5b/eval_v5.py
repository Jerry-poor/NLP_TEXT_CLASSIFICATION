import re
import json
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import csv
# ---------------- 1. Load Gold Data ----------------
df = pd.read_csv("preprocessed_dataset.csv", encoding="latin1")
_, test_df = train_test_split(df, test_size=0.3, random_state=42)
# Take first 1000 rows as evaluation gold answers (these are the original gold annotations)
gold_outputs = test_df['output'].iloc[:1000].tolist()

def is_json_complete(json_str):
    """
    Simple check whether a JSON string is complete (ends with '}')
    """
    return json_str.strip().endswith("}")

def evaluate_entities(gold_entities, pred_entities):
    """
    Compute entity-level evaluation (without token weighting):
      - TP: for each key (entity, order, label), the correct count is the minimum of the occurrence in gold and pred.
      - FP: extra predictions
      - FN: missing predictions
    """
    gold_counter = Counter((e['entity'], e['order'], e['label']) for e in gold_entities)
    pred_counter = Counter((e['entity'], e['order'], e['label']) for e in pred_entities)
    
    tp = sum(min(gold_counter[k], pred_counter.get(k, 0)) for k in gold_counter)
    fp = sum(pred_counter[k] - gold_counter.get(k, 0) for k in pred_counter if pred_counter[k] > gold_counter.get(k, 0))
    fn = sum(gold_counter[k] - pred_counter.get(k, 0) for k in gold_counter if gold_counter[k] > pred_counter.get(k, 0))
    
    return tp, fp, fn

# ---------------- 2. Initialize Counters ----------------
tp_total = 0
fp_total = 0
fn_total = 0

# For token-level
gold_token_total_global = 0
pred_token_total_global = 0
correct_token_total_global = 0

evaluated_sentence_count = 0  # Number of valid samples
correct_sentence_count = 0    # Number of sentences with complete match (based on multiset comparison)
total_gold_count = 0          # Total count of gold entities (not deduplicated)

# ---------------- 3. Read Model Output ----------------
with open("combined_eval_results_v5.txt", "r", encoding="latin1") as f:
    content = f.read()

# Split samples by the separator "------------------------"
samples = [s.strip() for s in content.split("------------------------") if s.strip()]

# ---------------- 4. Process Samples ----------------
gold_idx = 0  # index into gold_outputs
errors = []  # 用于存储错误预测的样本

for sample in samples:
    if gold_idx >= len(gold_outputs):
        break

    # Get current gold answer from gold_outputs
    try:
        gold_json = json.loads(gold_outputs[gold_idx])
    except json.JSONDecodeError:
        gold_idx += 1
        continue

    # Extract predicted JSON from sample using custom marker
    pred_match = re.search(r'Model Output:[\s\S]*?(\{"entities":\s*\[.*?\]\})', sample, re.DOTALL)
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

    # ---------------- Sentence-level Evaluation ----------------
    gold_counter = Counter((item['entity'], item['order'], item['label']) for item in gold_entities)
    pred_counter = Counter((item['entity'], item['order'], item['label']) for item in pred_entities)
    
    if gold_counter != pred_counter:  # 如果预测错误，记录到 errors 列表
        errors.append({
            "sentence": sample.split("Sentence (Input):")[1].split("\n")[0].strip(),
            "entity": gold_json,
            "model_output": pred_json
        })
    else:
        correct_sentence_count += 1
    # ---------------- Entity-level Evaluation ----------------
    tp, fp, fn = evaluate_entities(gold_entities, pred_entities)
    tp_total += tp
    fp_total += fp
    fn_total += fn
    total_gold_count += len(gold_entities)

    # ---------------- Token-level Evaluation ----------------
    # For token-level, we assume token count is computed by splitting the entity text by whitespace.
    # Gold token total: sum of token counts for each gold entity.
    gold_token_total_sample = sum(len(e['entity'].split()) for e in gold_entities)
    pred_token_total_sample = sum(len(e['entity'].split()) for e in pred_entities)
    
    # Build counters for gold and prediction using (entity, order, label) as key.
    gold_counter_tokens = Counter((e['entity'], e['order'], e['label']) for e in gold_entities)
    pred_counter_tokens = Counter((e['entity'], e['order'], e['label']) for e in pred_entities)
    # Record token length for each gold entity key (assume same for all occurrences)
    gold_token_length = {}
    for e in gold_entities:
        key = (e['entity'], e['order'], e['label'])
        gold_token_length[key] = len(e['entity'].split())
    
    # Correct token total for this sample: for each key, add min(count_gold, count_pred) * token_length
    correct_token_sample = sum(min(gold_counter_tokens[k], pred_counter_tokens.get(k, 0)) * gold_token_length[k] 
                                for k in gold_counter_tokens)
    
    gold_token_total_global += gold_token_total_sample
    pred_token_total_global += pred_token_total_sample
    correct_token_total_global += correct_token_sample

    gold_idx += 1

# ---------------- 5. Calculate Metrics ----------------
def safe_div(a, b):
    return a / b if b else 0

# Entity-level
accuracy_entity = safe_div(tp_total, tp_total + fp_total)
recall_entity = safe_div(tp_total, total_gold_count)
f1_entity = safe_div(2 * accuracy_entity * recall_entity, (accuracy_entity + recall_entity))

# Sentence-level
sentence_precision = safe_div(correct_sentence_count, evaluated_sentence_count)

# Token-level
token_precision = safe_div(correct_token_total_global, pred_token_total_global)
token_recall = safe_div(correct_token_total_global, gold_token_total_global)
f1_token = safe_div(2 * token_precision * token_recall, (token_precision + token_recall))

# ---------------- 6. Output Results ----------------
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

# ---------------- 7. Save Errors to CSV ----------------
with open("prediction_errors.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["sentence", "entity", "model_output"])
    writer.writeheader()
    for error in errors:
        writer.writerow({
            "sentence": error["sentence"],
            "entity": json.dumps(error["entity"], ensure_ascii=False),
            "model_output": json.dumps(error["model_output"], ensure_ascii=False)
        })

print(f"Saved {len(errors)} prediction errors to prediction_errors.csv")
