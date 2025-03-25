import os
import json
import pandas as pd
from openai import OpenAI

def main():
    # 加载 deepseek 配置文件
    with open('./api.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    apis = config.get('apis', [])
    prompt_template = config.get('prompt', '')
    # 检查 API 配置（此处选择第一个 API 配置）
    if not apis:
        print("没有找到 API 配置！")
        return
    api_info = apis[0]
    df = pd.read_csv('...../dataset/dataset0/abhinavwalia95.csv')
    
    # 定义参数
    batch_size = 50               # 每轮对话使用的训练数据条数
    num_conversations = 8         # 测试批次数
    training_rounds = 8           # 每个对话中训练轮次数量
    train_counter = 0             # 用于训练数据的全局计数器
    
    training_set = df.iloc[0:800].reset_index(drop=True)
    test_set = df.iloc[1000:1399].reset_index(drop=True)
    
    # 创建 OpenAI 客户端，关闭流式输入（stream=False）
    client = OpenAI(api_key=api_info["api_key"], base_url=api_info["base_url"])
    
    # 根据测试批次数循环构造对话并调用 API
    for conv_idx in range(num_conversations):
        conversation = []
        # 构造训练轮次对话
        for round_idx in range(training_rounds):
            start_idx = train_counter
            end_idx = train_counter + batch_size
            
            # 若超出训练数据范围则循环使用（取余）
            if end_idx > len(training_set):
                sentences = []
                entities = []
                for i in range(batch_size):
                    idx = (train_counter + i) % len(training_set)
                    sentences.append(training_set.loc[idx, 'sentence'])
                    entities.append(training_set.loc[idx, 'entities'])
            else:
                # 使用 iloc 保证切片正确（end_idx 不包含在内）
                sentences = training_set.iloc[start_idx:end_idx]['sentence'].tolist()
                entities = training_set.iloc[start_idx:end_idx]['entities'].tolist()
            train_counter += batch_size
            
            # 第一轮用户消息附带 prompt 模板，其余轮仅传入句子
            if round_idx == 0:
                user_content = prompt_template + "\n" + "\n".join(sentences)
            else:
                user_content = "\n".join(sentences)
            assistant_content = "\n".join(entities)
            
            conversation.append({
                "role": "user",
                "content": user_content
            })
            conversation.append({
                "role": "assistant",
                "content": assistant_content
            })
        
        # 构造测试轮：只发送用户消息，期待 API 返回结果
        test_start = conv_idx * batch_size
        test_end = (conv_idx + 1) * batch_size
        if test_end > len(test_set):
            test_end = len(test_set)
        test_sentences = test_set.iloc[test_start:test_end]['sentence'].tolist()
        user_content_test = "\n".join(test_sentences)
        
        conversation.append({
            "role": "user",
            "content": user_content_test
        })
        
        # 构造完整请求体
        complete_query = {
            "messages": conversation
        }
        
        try:
            # 调用 API，关闭流式输入
            response = client.chat.completions.create(
                model=api_info["model"],
                messages=conversation,
                stream=False
            )
            # 获取返回的响应内容（注意处理可能存在的空行或 SSE keep-alive 注释）
            result_content = response.choices[0].message.content.strip()
            
            # 构造最终输出，包含请求内容和 API 返回结果
            output = {
                "request": complete_query,
                "response": result_content
            }
            output_filename = f"test_batch{conv_idx + 1}.txt"
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(json.dumps(output, ensure_ascii=False, indent=4))
            print(f"对话 {conv_idx + 1} 的请求及响应已存储到 {output_filename}")
        except Exception as e:
            print(f"对话 {conv_idx + 1} 调用 API 时出错: {e}")

if __name__ == '__main__':
    main()
