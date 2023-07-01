from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import time
import torch
import pandas as pd
import ast
from keras.layers import Dropout
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import backend as K
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import kendalltau
from keras.callbacks import TensorBoard, Callback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Reshape

def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def train(bz, delta, p, m, n, e, lamuda):
    set_seeds(0)
    df = pd.read_csv('data/data_No.csv')
    df['embedding'] = df['embedding'].apply(ast.literal_eval)
    max_length = max(df['embedding'].apply(len))
    df['embedding'] = df['embedding'].apply(lambda x: x + [0] * (max_length - len(x)))
    embedding_matrix = np.array(df['embedding'].tolist())
    # MODIFIED 
    data = embedding_matrix.reshape((len(embedding_matrix), 60, 4, 1))  # reshape data to be (samples, height, width, channels)
    DNA = data
    labels = np.array(df['deletions'].tolist())
    x_train, x_test, y_train, y_test = train_test_split(DNA, labels, test_size = 0.20, random_state = 0)
    x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size = 0.5, random_state = 0)
    model = Sequential()
    filters = 256
    kernel_size = (3, 3) 
    input_shape = (60,4, 1) 
    model.add(Conv2D(filters, kernel_size, input_shape=input_shape,))
    model.add(Conv2D(filters, (2, 2), input_shape=input_shape)) 
    model.add(Flatten())
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(1)) 
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    model.fit(x_train, y_train, epochs=e,validation_split=0.2,
                        verbose=1, batch_size=bz,callbacks=[ ])
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    y_pred = np.squeeze(y_pred) 
    
    print("Test: ",y_test)
    print("Pred: ",y_pred)
    print("R2 Score:", r2)
    
    tau, p_value = kendalltau(y_test, y_pred)
    print("Kendall's Tau:", tau)
    
    corr_matrix = np.corrcoef(y_test, y_pred)
    pearson_coef = corr_matrix[0, 1]
    print("Pearson:", pearson_coef)
    comment = "(" + "del fre" + ")"
    name = 'model' + str(pearson_coef) + comment + ".h5"
    # 保存整个模型
    print(name," is saved")
    model.save(name)
#modified
train(512, 0.001, 0.01, 0, 30, 100, 0.01)
