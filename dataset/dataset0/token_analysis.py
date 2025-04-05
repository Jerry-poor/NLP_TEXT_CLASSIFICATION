import pandas as pd
import json

df = pd.read_csv("answer_text.csv", encoding="latin1")

def count_tokens(text):
    #只是预估token，所以不使用模型专用的分词器
    tokens = text.split()
    return len(tokens)

def process_text(text_str):
    """
    尝试将 text_str 解析为 JSON，如果成功则对其中的 sentence 字段分词；
    如果解析失败，则直接对 text_str 分词统计。
    """
    try:
        data = json.loads(text_str)
        sentence = data.get("sentence", text_str)
    except Exception:
        sentence = text_str
    return count_tokens(sentence)

df["token_count"] = df["text"].apply(process_text)
df.to_csv("token_count_analysis.csv", index=False, encoding="latin1")
print("Token count analysis saved to token_count_analysis.csv")
