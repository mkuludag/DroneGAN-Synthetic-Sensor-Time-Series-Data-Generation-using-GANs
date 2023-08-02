# file to paste the various models to be used in main progrma, less clutter

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Reshape, Dropout, Conv2D, MaxPooling2D, BatchNormalization, ConvLSTM2D
import pdb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.utils import plot_model

def first_model(input_shape):
    #input_shape = (K, data.shape[1])
    model = Sequential()
    model.add(Conv1D(30, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Conv1D(30, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    #model.add(Dropout(0.5)) # for regularization
    model.add(Dense(1, activation='sigmoid'))

    return model

# def first_model(input_shape):
#     model = Sequential()
#     model.add(ConvLSTM2D(512, kernel_size=(3,3), activation='relu', input_shape=input_shape, return_sequences=True, data_format='channels_first'))
#     model.add(BatchNormalization())

#     model.add(ConvLSTM2D(256, kernel_size=(3,3), activation='relu', input_shape=input_shape, return_sequences=True, data_format='channels_first'))
#     model.add(BatchNormalization())

#     model.add(ConvLSTM2D(128, kernel_size=(3,3), activation='relu', input_shape=input_shape, return_sequences=True, data_format='channels_first'))
#     model.add(BatchNormalization())

#     model.add(ConvLSTM2D(64, kernel_size=(3,3), activation='relu', input_shape=input_shape, return_sequences=True, data_format='channels_first'))
#     model.add(BatchNormalization())

#     model.add(ConvLSTM2D(32, kernel_size=(2,2), activation='relu', input_shape=input_shape, return_sequences=True, data_format='channels_first'))
    
#     model.add(Flatten())

#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(500, activation='sigmoid'))
#     model.add(Reshape((5, 10, 10)))

#     return model