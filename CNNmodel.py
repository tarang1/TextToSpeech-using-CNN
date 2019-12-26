# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 23:32:41 2018

@author: Tarang
"""

from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

seed = 128
rng = np.random.RandomState(seed)

train = pd.read_csv("../MachineLearning3/handwritten_data_785.csv",encoding = 'utf8')
train_x = train.iloc[:,1:785].values
train_y = keras.utils.np_utils.to_categorical(train.iloc[:,0].values)


train_x_temp = train_x.reshape(-1,28,28,1)
input_shape = (784,)
input_reshape = (28, 28, 1)

conv_num_filters = 5
conv_filter_size = 5

pool_size = (2, 2)

hidden_num_units = 50
output_num_units = 26

epochs = 5
batch_size = 128

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D 
from keras.layers import InputLayer
from keras.layers import Flatten


model = Sequential([
 InputLayer(input_shape=input_reshape),

 Convolution2D(25, 5, 5, activation='relu'),
 MaxPooling2D(pool_size=pool_size),

 Convolution2D(25, 5, 5, activation='relu'),
 MaxPooling2D(pool_size=pool_size),

 Convolution2D(25, 4, 4, activation='relu'),

 Flatten(),

 Dense(output_dim=hidden_num_units, activation='relu'),

 Dense(output_dim=output_num_units, input_dim=hidden_num_units, activation='softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

trained_model_conv = model.fit(train_x_temp, train_y, nb_epoch=epochs, batch_size=batch_size)

img = Image.open('../MachineLearning4/rsz_test1.jpg').convert('L')
arr = np.array(img)
b = 1 - arr
br = np.expand_dims(b,axis = 2)
brr = np.expand_dims(br,axis = 0)
pred = model.predict_classes(brr)
A = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
text = A[pred[0]]
f = open("text.txt","w")
f.write(text)
f.close()
