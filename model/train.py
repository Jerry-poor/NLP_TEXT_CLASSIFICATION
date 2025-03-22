import pandas as pd
#封装原有的process_data函数并更名为train_data的函数
#这函数要求接收一个df，并修改内容，从只关注对应实体以及标签修改成需要关注对应的位置变成将位置也加入训练
#同时新增一个map参数，用于传入映射函数，该映射函数不会存在于train.py，而是在模型调用的主代码区域
def train_data(df, map):
    texts = []
    labels = []
    for index, row in df.iterrows():
        text = row['sentence']
        texts.append(text)
        entities = row['entities']
        entity_labels = []
        for entity in eval(entities):
            entity_labels.append([entity[0], entity[1][0], entity[1][1] + 1, map(entity[2])])
        labels.append(entity_labels)
    return texts, labels
