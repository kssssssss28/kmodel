
import torch
import pandas as pd
import ast
import tensorflow as tf
import numpy as np
import os

df = pd.read_csv('data/modified_with_embedding.csv')

df['embedding'] = df['embedding'].apply(ast.literal_eval)

max_length = max(df['embedding'].apply(len))
df['embedding'] = df['embedding'].apply(lambda x: x + [0] * (max_length - len(x)))
embedding_matrix = np.array(df['embedding'].tolist())


data= embedding_matrix
data = embedding_matrix.reshape((len(embedding_matrix), 60, 6, 1))  # reshape data to be (samples, height, width, channels)
    
# 解析文本数据并转换为浮点数格式的多维数组

def addLabel(index, type):
    insertion = df["insertions"][index]
    if np.isnan(insertion):
        insertion = 0
        
    deletion = df["deletions"][index]
    if np.isnan(deletion):
        deletion = 0
        
    singleInsertion = df["delfreq"][index]
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
    labels = np.array(loadData(type=2))
    DNA = data
    print(labels)
    return DNA, labels