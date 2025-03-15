import pandas as pd
import ast  
# 用于将字符串表示的列表转换为列表对象
df = pd.read_csv("output_sentences_with_entities.csv")

unique_labels = set()

for entities in df['entities']:
    # 将字符串形式的实体列表转换为真正的列表
    entity_list = ast.literal_eval(entities)
    
    # 提取实体标签并添加到 set 中（set 会自动去重）
    for entity in entity_list:
        unique_labels.add(entity[2])  
print("Unique entity labels:", unique_labels)
