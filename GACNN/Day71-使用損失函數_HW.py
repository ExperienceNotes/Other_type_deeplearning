# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 13:32:12 2020

@author: user
"""

import keras
from keras.datasets import cifar10
import numpy as np
from keras.utils import to_categorical
from keras.layers import Dense,Input,Conv2D,MaxPooling2D,Flatten
from keras.models import Model
import matplotlib.pyplot as plt
np.random.seed(10)
#np.random.seed(10)的作用：使得隨機數據可預測


(x_train,y_train),(x_test,y_test)=cifar10.load_data()

print('train data:{} label:{}'.format(x_train.shape,y_train.shape))
print('test data:{} label:{}'.format(x_test.shape,y_test.shape))

x_train_normal = 1-x_train.astype('float32')/127
x_test_normal = 1-x_test.astype('float32')/127

y_train_OneHot = to_categorical(y_train)
y_test_OneHot = to_categorical(y_test)

def model_net():
    Inputs = Input(name = 'Input',shape = (32,32,3))
    x = Conv2D(filters = 32,kernel_size = (3,3),activation = 'relu',padding = 'same')(Inputs)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Conv2D(filters = 64,kernel_size = (3,3),activation = 'relu',padding = 'same')(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Conv2D(filters = 128,kernel_size = (3,3),activation = 'relu',padding = 'same')(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Conv2D(filters = 256,kernel_size = (3,3),activation = 'relu',padding = 'same')(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Flatten()(x)
    x = Dense(512,activation = 'relu')(x)
    x = Dense(256,activation = 'relu')(x)
    x = Dense(10,activation = 'sigmoid')(x)
    models = Model(Inputs, x)
    
    print(models.summary())
    return models
    
    
model_net = model_net()
model_net.compile(loss = 'binary_crossentropy',optimizer='Adam', metrics=['accuracy'])
train_history = model_net.fit(x = x_train_normal,y = y_train_OneHot,
              validation_split = 0.2,epochs = 12,batch_size = 128,verbose = 1)

def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train_acc', 'val_acc'], loc='upper left')
    plt.show()
    
    
show_train_history('accuracy','val_accuracy')

show_train_history('loss','val_loss')    
    
    