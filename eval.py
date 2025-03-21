'''
评估代码。要计算三个指标：精确度，召回率和f1-score,分为三种级别，句子级别，实体级别和token级别。
实体级别以预测准确实体位置和标签为一次正确预测。token级则计算预测正确的token（一个实体可能包含多个token，一个句子可能包含多个或者零个实体）。
三种级别都需要计算三种指标。如果句子级别预测正确，则添加进True_df.csv中，作为可信数据集.
'''
import os
import pandas as pd
def evaluate_model(nlp, texts, labels, path, model_name):
    sentence_total = 0
    sentence_correct = 0
    entity_tp = 0
    entity_fp = 0
    entity_fn = 0
    token_correct = 0
    token_pred_total = 0
    token_true_total = 0
    true_sentences = []
    for doc_text, true_label in zip(texts, labels):
        sentence_total += 1
        doc = nlp(doc_text)
        pred_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        true_entities = []
        for entity in true_label:
            if entity[2] == 'O':
                continue
            true_entities.append((entity[1][0], entity[1][1] + 1, entity[2]))
        if set(pred_entities) == set(true_entities):
            sentence_correct += 1
            true_sentences.append(doc_text)
        matched_true = set()
        for pred in pred_entities:
            if pred in true_entities:
                entity_tp += 1
                matched_true.add(pred)
            else:
                entity_fp += 1
        for true_ent in true_entities:
            if true_ent not in matched_true:
                entity_fn += 1
        true_token_labels = ["O"] * len(doc)
        pred_token_labels = ["O"] * len(doc)
        for (start, end, label) in true_entities:
            span = doc.char_span(start, end, alignment_mode="contract")
            if span is not None:
                for token in span:
                    true_token_labels[token.i] = label
        for ent in doc.ents:
            for token in ent:
                pred_token_labels[token.i] = ent.label_
        for i in range(len(doc)):
            if pred_token_labels[i] == true_token_labels[i] and true_token_labels[i] != "O":
                token_correct += 1
        token_pred_total += sum(1 for lab in pred_token_labels if lab != "O")
        token_true_total += sum(1 for lab in true_token_labels if lab != "O")
    sentence_precision = sentence_correct / sentence_total if sentence_total > 0 else 0
    sentence_recall = sentence_correct / sentence_total if sentence_total > 0 else 0
    sentence_f1 = sentence_precision
    entity_precision = entity_tp / (entity_tp + entity_fp) if (entity_tp + entity_fp) > 0 else 0
    entity_recall = entity_tp / (entity_tp + entity_fn) if (entity_tp + entity_fn) > 0 else 0
    entity_f1 = 2 * entity_precision * entity_recall / (entity_precision + entity_recall) if (entity_precision + entity_recall) > 0 else 0
    token_precision = token_correct / token_pred_total if token_pred_total > 0 else 0
    token_recall = token_correct / token_true_total if token_true_total > 0 else 0
    token_f1 = 2 * token_precision * token_recall / (token_precision + token_recall) if (token_precision + token_recall) > 0 else 0
    print("Sentence-level Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%".format(sentence_precision*100, sentence_recall*100, sentence_f1*100))
    print("Entity-level Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%".format(entity_precision*100, entity_recall*100, entity_f1*100))
    print("Token-level Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%".format(token_precision*100, token_recall*100, token_f1*100))
    true_df = pd.DataFrame({'sentence': true_sentences})
    true_df.to_csv(os.path.join(path, 'True_df.csv'), index=False)
    # 添加日志到 path 文件夹的 log 子目录中
    log_dir = os.path.join(path, "log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    base_log = os.path.join(log_dir, "log.txt")
    if not os.path.exists(base_log):
        log_file = base_log
    else:
        i = 1
        while True:
            candidate = os.path.join(log_dir, f"log{i}.txt")
            if not os.path.exists(candidate):
                log_file = candidate
                break
            i += 1
    with open(log_file, "w") as f:
        f.write(model_name)
