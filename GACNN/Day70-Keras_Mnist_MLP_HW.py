# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 20:47:44 2020

@author: user
"""


import keras
from keras.layers import Dense,Input
from keras.datasets.mnist import load_data
from keras.utils import to_categorical
from keras.models import Model
import matplotlib.pyplot as plt
(x_train,y_train),(x_test,y_test) = load_data()
x_train = x_train.reshape(60000,784).astype('float32')
x_test = x_test.reshape(10000,784).astype('float32')

x_train_normal = x_train / 255
x_test_normal = x_test / 255

y_train_OneHot = to_categorical(y_train,10)
y_test_OneHot = to_categorical(y_test, 10)

inputs = Input(name='input', shape=(784,))
x = Dense(256, activation='relu', name='Dense_1')(inputs)
x = Dense(128, activation='relu', name='Dense_2')(x)
x = Dense(64, activation='relu', name='Dense_3')(x)
x = Dense(10, activation='softmax', name='Dense_4')(x)

models = Model(inputs, x)

models.summary()
models.compile(loss='categorical_crossentropy', 
               optimizer='adam', 
               metrics=['accuracy'])

train_history = models.fit(x = x_train_normal,
                           y = y_train_OneHot,validation_split=0.2,
                           epochs = 10,batch_size = 32,verbose=1)

def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    
show_train_history(train_history,'accuracy','val_accuracy')
show_train_history(train_history,'loss','val_loss')
scores = models.evaluate(x_test_normal, y_test_OneHot)
print()
print('accuracy=',scores[1])

