# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 20:14:52 2020

@author: user
"""
from keras.datasets import cifar10
from GenerationGA import GenerationalGA
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
from keras.utils import to_categorical
y_train, y_test = to_categorical(y_train), to_categorical(y_test)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

train_size = len(x_train)
test_size = len(x_test)
g = GenerationalGA(
    x_train=x_train[:train_size],
    y_train=y_train[:train_size],
    x_test=x_test[:test_size],
    y_test=y_test[:test_size],
    pop_size=20,
    r_mutation=0.1,
    p_crossover=0,  # no use
    p_mutation=0.2,
    max_iter=10,
    min_fitness=0.95,
    batch_size=5000,
    elite_num=0,  # no use
    mating_pool_size=0,  # no use
)
g.run()

