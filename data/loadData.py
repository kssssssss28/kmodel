
import torch
import pandas as pd
import ast
import tensorflow as tf
import numpy as np
import os

current_path = os.getcwd()
print("当前路径：", current_path)
# 读取CSV文件
# 读取CSV文件

df = pd.read_csv('data/modified_with_embedding.csv')

# 将字符串解析为嵌套列表
df['embedding'] = df['embedding'].apply(ast.literal_eval)

# 获取最大的子列表长度
max_length = max(df['embedding'].apply(len))

# 将嵌套列表进行填充，使所有子列表具有相同的长度
df['embedding'] = df['embedding'].apply(lambda x: x + [0] * (max_length - len(x)))

# 将嵌套列表转换为NumPy数组
embedding_matrix = np.array(df['embedding'].tolist())


data= embedding_matrix
# 解析文本数据并转换为浮点数格式的多维数组

def addLabel(index, type):
    insertion = df["insertions"][index]
    if np.isnan(insertion):
        insertion = 0
    deletion = df["deletions"][index]
    if np.isnan(deletion):
        deletion = 0
    singleInsertion = df["oneinsertion"][index]
    if np.isnan(singleInsertion):
        singleInsertion = 0
    singleDeletion = df["onedeletion"][index]
    if np.isnan(singleDeletion):
        singleDeletion = 0
    total_out = df["total_out"][index]
    if np.isnan(total_out):
        total_out = 0
        
    if type==0:
        temp = [insertion, deletion, singleInsertion, singleDeletion]
    elif type==1:
        temp = [deletion]
    elif type==2:
        temp = [singleInsertion]
    elif type==3:
        temp = [singleDeletion]
    elif type==4:
        temp = [insertion]
    temp = [float(value) for value in temp] 
    return temp


def loadData(type):
    label = []
    total = len(df["genename"])
    random_array = np.random.choice(np.arange(0, total), size=total, replace=False)
    for i in range(len(df["genename"])):
        label.append(addLabel(random_array[i], type))  # 传递type参数给addLabel函数
    return label

      


def get_data_split(type):
    # 转换样本数据为NumPy数组
    DNA = data
    

        
    # 转换标签数据为NumPy数组
    labels = np.array(loadData(type))

    # 训练集：前 80% 的数据
    train_indices = slice(0, 1287)
    train_data = DNA[train_indices]
    train_labels = labels[train_indices]

    # 验证集：接下来的 10% 的数据
    valid_indices = slice(1287, 1447)
    valid_data = DNA[valid_indices]
    valid_labels = labels[valid_indices]

    # 测试集：剩余的 10% 的数据
    test_indices = slice(1447, None)
    test_data = DNA[test_indices]
    test_labels = labels[test_indices]
    return DNA, labels

    return (train_data, train_labels),(test_data, test_labels),(valid_data,valid_labels)