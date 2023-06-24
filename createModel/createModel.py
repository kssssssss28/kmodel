import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten,MaxPooling2D
from tensorflow.keras.layers import Conv2D

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Bidirectional, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Bidirectional, Dense, Reshape


def createSingleModel(lamuda):
    model = Sequential()
    filters = 64
    kernel_size = (3, 6) 
    input_shape = (60, 6, 1) 

    model.add(Conv2D(filters, kernel_size, input_shape=input_shape,))

    pool_size = (3, 1)
    stride = (1, 1)

    model.add(MaxPooling2D(pool_size=pool_size, strides=stride))

    model.add(Flatten())


    model.add(Dense(50, activation='tanh')) 
    model.add(Dense(50, activation='tanh')) 


    model.add(Dense(4)) 
    
    
    return model
