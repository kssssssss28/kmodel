import tensorflow as tf
import numpy as np
import pandas as pd
keydf = pd.read_csv('./sproutData.csv')
splitedArr = []
keydf.head()

def positional_encoding(sequence_length, d_model):
    # 生成位置编码矩阵
    position_encodings = np.zeros((sequence_length, d_model))
    for pos in range(sequence_length):
        for i in range(d_model):
            if i % 2 == 0:
                position_encodings[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            else:
                position_encodings[pos, i] = np.cos(pos / (10000 ** ((i - 1) / d_model)))

    return tf.cast(position_encodings, dtype=tf.float32)

# 设置序列长度和模型维度
sequence_length =60
d_model = 1

# 生成位置编码
position_encodings = positional_encoding(sequence_length, d_model).numpy()





def seqSplit(file, window=3, step=1):
  seq  = file['refseq']
  seqArr = seq.tolist()
  kmer_dict = {}
  result = {'splitedArr': [], 'splitedArrString': []}
  for seq in seqArr:
      k_mers = [seq[i:i+window] for i in range(0, len(seq)-window+1, step)]
      result['splitedArr'].append(k_mers)
      result['splitedArrString'].append(' '.join(k_mers))
  return result


def addNewSplitedCol(window, step):
    splitedArr = seqSplit(keydf, window, step)
    keydf['splitedSeq'] = splitedArr['splitedArr']
    keydf['splitedArrString'] = splitedArr['splitedArrString']
    keydf.to_csv('modified.csv', index=False)

encoding_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}


def getPAM():
    data = pd.read_csv('modified.csv')
    seqData = data["refseq"]
    PAM = []
    encoding_dict = {"A": [1, 0, 0, 0], "C": [0, 1, 0, 0], "G": [0, 0, 1, 0], "T": [0, 0, 0, 1]}  # 定义编码字典
    embedding = []
    position_encodings = positional_encoding(sequence_length, d_model).numpy()  # 假设已经获取到位置编码

    for index, seq in enumerate(seqData):
        temp = seq[33:36]
        PAM.append(temp)

        encoded_sequence = []
        for i, base in enumerate(seq):
            encoded_base = []
            encoded_base.extend(encoding_dict[base])
            #encoded_base.append(position_encodings[i][0])
            # if i > 30 and i < 36:
            #     # 执行相应的操作
            #     encoded_base.append(0)
            # else:
            #     # 执行其他操作
            #     encoded_base.append(0)

            encoded_sequence.append(encoded_base)


        # 添加位置编码


        embedding.append(encoded_sequence)

    data['PAM'] = PAM
    data['embedding'] = embedding
    data.to_csv('modified_with_embedding_no.csv', index=False)
    return data

addNewSplitedCol(3,3)
getPAM()
