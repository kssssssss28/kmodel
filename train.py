from data.loadData import get_data_split
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from createModel.createModel import createSingleModel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import backend as K
import keras
from sklearn.metrics import r2_score, mean_squared_error

from keras.callbacks import TensorBoard
from keras.callbacks import Callback

 
        
def train(bz, delta, p, m, n, e, lamuda):
    
    model = createSingleModel(lamuda)
    model.summary()
    ###(train_data, train_labels), (test_data, test_labels) ,(valid_data,valid_labels)= 
    DNA, labels= get_data_split(0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=p, patience=m)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=n, verbose=1)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    
    
    
    x_train, x_test, y_train, y_test = train_test_split(DNA, labels, test_size = 0.20, random_state = 0)
    x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size = 0.5, random_state = 0)
    model.fit(x_train, y_train, epochs=e,
                        validation_data=(x_valid, y_valid),
                        verbose=1, batch_size=bz,
                        callbacks=[TensorBoard(log_dir='./logs', histogram_freq=1),reduce_lr, early_stopping, ])


    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    print(y_test,y_pred)
    print("R2 Score:", r2)


train(32, 0.0001, 0.1, 0, 30, 20, 0.01)
