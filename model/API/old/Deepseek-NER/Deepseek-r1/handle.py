# handle.py
import os
import json
import pandas as pd

def main():
    # 读取 CSV 数据，要求 CSV 包含 "sentence" 与 "entities" 两列
    df = pd.read_csv('../dataset0/abhinavwalia95.csv')
    
    # 前800行作为训练示例
    training_set = df.head(800)
    # 800~1000行作为待处理数据
    inference_set = df.iloc[800:1000].copy()
    
    # 为待处理数据增加两列，初始设置 finishment=0 和 api_result=None
    inference_set['finishment'] = 0
    inference_set['api_result'] = None
    
    batch_size = 200
    # 划分批次，保持与 api.py 中相同的逻辑
    batch_indices = [inference_set.index[i:i+batch_size] for i in range(0, len(inference_set), batch_size)]
    
    result_folder = './result'
    
    # 对于每个批次，读取对应的 txt 文件 ds{batch_num}.txt
    for i, indices in enumerate(batch_indices):
        file_path = os.path.join(result_folder, f'ds{i}.txt')
        if not os.path.exists(file_path):
            print(f"文件 {file_path} 不存在，跳过批次 {i}")
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            result = f.read()
        # 尝试解析返回的结果为 JSON 格式
        try:
            parsed_result = json.loads(result)
        except Exception as e:
            print(f"批次 {i} 解析JSON失败: {e}")
            parsed_result = {}
        # 设定 temp：存储当前批次起始行的前一行索引
        batch_start = min(indices)
        temp = batch_start - 1
        # 遍历返回的字典，根据返回的 "sentenceX" 键计算原始行号
        for key, value in parsed_result.items():
            try:
                sentence_num = int(key.replace("sentence", ""))
            except Exception as e:
                print(f"无法解析句子编号 {key}: {e}")
                continue
            # 计算原始行号：original_index = temp + sentence_num + 1
            original_index = temp + sentence_num + 1
            # 对返回的 value 进行简单处理（例如去除前后空白）
            processed_value = value.strip() if isinstance(value, str) else value
            inference_set.at[original_index, 'api_result'] = processed_value
            inference_set.at[original_index, 'finishment'] = 1
    
    # 对训练集，由于它们用于提供示例，直接将原有实体信息复制到 api_result，标记 finishment 为 1
    training_set = training_set.copy()
    training_set['finishment'] = 1
    training_set['api_result'] = training_set['entities']
    
    # 合并训练集和待处理数据，输出到 new_data.csv
    new_df = pd.concat([training_set, inference_set], ignore_index=True)
    new_df.to_csv('new_data.csv', index=False)
    print("处理完成，新数据保存在 new_data.csv 中。")

if __name__ == '__main__':
    main()
