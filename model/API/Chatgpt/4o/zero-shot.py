import os
import json
import pandas as pd
import concurrent.futures
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

# 加载 deepseek 配置文件
with open('./api.json', 'r', encoding='latin1') as f:
    config = json.load(f)

apis = config.get('apis', [])
# 从配置中也可获取 prompt 模板，如有需要；此处直接使用下列 prompt 文本
prompt_template = (
    "You are an NLP‑savvy research assistant. Using the training examples below, perform Named Entity Recognition (NER) on the *input text*."
    "【Task】Identify every entity in the text according to the predefined label list and record the order in which each entity appears in the entire text."
    "【Entity labels】per = Person; org = Organization; gpe = Geopolitical Entity; geo = Geographical Entity; tim = Time indicator; art = Artifact; eve = Event; nat = Natural Phenomenon."
    "【Output format (must be followed exactly)】- Return one valid JSON object (latin1‑encoded)."
    " Structure must be: {\"entities\":[{\"entity\":\"EntityName\",\"order\":0,\"label\":\"per\"}, ...]}."
    " * entities: array, entities listed in the order they appear in the text."
    " * entity: the exact surface form from the text (case‑preserved), wrapped in double quotes."
    " * order: 0 for the first occurrence of that entity in the whole text, 1 for the second, etc."
    " * label: one of \"per\",\"org\",\"gpe\",\"geo\",\"tim\",\"art\",\"eve\",\"nat\"."
    " No spaces, line breaks, or explanatory text outside string values."
    " If no entity is found, return {\"entities\":[]}."
    " If you cannot fully satisfy the above, return {}."
    "【Steps】1. Read the training examples to understand the annotation style and label usage."
    " 2. Perform NER on the *input text* and build the entities array."
    " 3. Output the minimized JSON exactly as specified."
    "【Training examples】<TRAINING_EXAMPLES>"
    "【Input text】<INPUT_TEXT>"
)

# Azure AI 项目连接字符串与认证，请替换为实际值
project_connection_string = "<your-connection-string-goes-here>"
project = AIProjectClient.from_connection_string(
    conn_str=project_connection_string, credential=DefaultAzureCredential()
)
chat = project.inference.get_chat_completions_client()

def call_api(text):
    try:
        response = chat.complete(
            model=apis[0]["model"],  # 这里取配置中第一个 API 的 model，可根据需要调整
            messages=[
                {
                    "role": "system",
                    "content": prompt_template
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )
        # 直接获取返回字符串
        result = response.choices[0].message.content
        return result, True
    except Exception as e:
        print(f"调用 API {apis[0]['name']} 异常: {e}")
        return None, False

def process_batch(batch_sentences, training_context):
    """
    构造当前批次输入（将训练示例与批次待处理句子拼接），调用 API 并返回结果字符串。
    """
    batch_input = "待处理句子：\n" + "\n".join(batch_sentences)
    combined_text = training_context + "\n" + batch_input
    result, success = call_api(combined_text)
    if success:
        return result, 0
    return None, 1

def match(orig_df, api_dict, batch_start):
    """
    对当前批次原始数据(orig_df)与 API 返回的字典(api_dict)进行匹配：
      - 对于 orig_df 中每一行，提取句子前10个字符作为 key；
      - 若该 key 存在于 api_dict 且对应列表不为空，则取该列表中的第一个注释更新该行的 api_result，并将 finishment 置为0，
        同时从列表中移除该注释。
    返回更新后的 orig_df 以及本批次处理结束的行号 end_row，
      计算公式为：end_row = batch_start + match_count - 1，其中 match_count 为本批次匹配成功的行数。
    """
    match_count = 0
    for idx, row in orig_df.iterrows():
        key = row["sentence"][:10]
        if key in api_dict and len(api_dict[key]) > 0:
            annotation = api_dict[key].pop(0)
            orig_df.at[idx, "api_result"] = annotation
            orig_df.at[idx, "finishment"] = 0
            match_count += 1
    end_row = batch_start + match_count - 1 if match_count > 0 else batch_start - 1
    return orig_df, end_row

def main():
    # 读取 CSV 数据，要求 CSV 包含 "sentence" 与 "entities" 两列
    df = pd.read_csv('../dataset0/abhinavwalia95.csv', encoding='latin1')
    
    # 前800行作为训练示例；从第1000行开始作为待处理数据
    training_set = df.head(800)
    inference_set = df.iloc[1001:1200].copy()
    # 初始化待处理数据的 finishment 为默认值1，api_result 为空
    inference_set['finishment'] = 1
    inference_set['api_result'] = None
    
    # 构造训练示例文本，每行格式为 "句子：{sentence} 实体：{entities}"
    training_examples = []
    for idx, row in training_set.iterrows():
        training_examples.append(f"句子：{row['sentence']} 实体：{row['entities']}")
    training_context = "训练示例：\n" + "\n".join(training_examples)
    
    batch_size = 200
    # 当前批次起始行号，假设 inference_set 的索引为连续整数
    current_start = inference_set.index[0]
    max_index = inference_set.index[-1]
    batch_count = 0
    
    # 清空或创建调试文件
    debug_filename = "./result.txt"
    with open(debug_filename, "w", encoding="utf-8") as debug_file:
        debug_file.write("API 调试结果记录：\n\n")
    
    # 逐批处理待处理数据，动态更新 inference_set 中对应行的 api_result 和 finishment
    while current_start <= max_index:
        batch_end = current_start + batch_size - 1
        if batch_end > max_index:
            batch_end = max_index
        batch_df = inference_set.loc[current_start:batch_end].copy()
        batch_sentences = batch_df['sentence'].tolist()
        
        # 调用 API 处理当前批次
        result, fin_flag = process_batch(batch_sentences, training_context)
        
        # 将 API 返回结果追加写入调试文件
        with open(debug_filename, "a", encoding="utf-8") as debug_file:
            debug_file.write(f"批次 {batch_count} 返回结果:\n{result}\n\n")
        
        if result is None:
            print(f"批次 {batch_count} API调用失败，跳过本批次")
            current_start = batch_end + 1
            batch_count += 1
            continue

        # 解析 API 返回结果为字典，要求键为句子前10字符，值为实体标注列表
        try:
            parsed_result = json.loads(result)
        except Exception as e:
            print(f"批次 {batch_count} JSON解析错误: {e}")
            parsed_result = {}
        
        # 调用 match 函数对当前批次进行比对，更新 batch_df，并获得本批次结束行号
        updated_batch_df, end_row = match(batch_df, parsed_result, current_start)
        inference_set.update(updated_batch_df)
        
        # 下一个批次的起始行
