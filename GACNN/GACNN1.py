# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 19:53:04 2020

@author: user
"""
from keras.datasets.mnist import load_data
from keras import backend as k
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.vis_utils import plot_model
from numpy.random import randint
from random import choice
from numpy.random import uniform
import numpy as np
from sys import exit
def mnist(show_examples=False):
    i_r, i_c, n_c = 28, 28, 10  # Image row, columns and class number 
    (x_tr, y_tr), (x_te, y_te) = load_data()  # Load mnist data
    if k.image_data_format() == 'channels_first':
        x_tr = x_tr.reshape(x_tr.shape[0], 1, i_r, i_c)
        x_te = x_te.reshape(x_te.shape[0], 1, i_r, i_c)
        i_sh = (1, i_r, i_c)
    else:
        x_tr = x_tr.reshape(x_tr.shape[0], i_r, i_c, 1)
        x_te = x_te.reshape(x_te.shape[0], i_r, i_c, 1)
        i_sh = (i_r, i_c, 1)
    x_tr = x_tr.astype('float32')
    x_te = x_te.astype('float32')
    y_tr = to_categorical(y_tr, n_c)
    y_te = to_categorical(y_te, n_c)
    x_tr /= 255  # Normalize training images
    x_te /= 255  # Normalize test     images
    if show_examples:
        print('X_tr:', x_tr.shape)
        print(x_tr.shape[0], 'train samples')
        print(x_te.shape[0], 'test  samples')
    return x_tr, x_te, y_tr, y_te, i_sh

class Net:
    def __init__(self):
        self.ep = randint(1,3)
        self.f1 = randint(30,34)
        self.f2 = randint(62,66)
        self.u1 = randint(126,130)
        self.k1 = choice([(3,3),(5,5)])
        self.k2 = choice([(3,3),(5,5)])
        self.d1 = choice([0.25,0.5])
        self.d2 = choice([0.25,0.5])
        self.a1 = 'relu'
        self.a2 = 'relu'
        self.a3 = 'relu'
        self.a4 = 'softmax'
        self.lf = 'categorical_crossentropy'
        self.op = 'adadelta'
        self.ac = 0
        
        self.particle = 20
        self.dim = 10
        self.Xmax = 100
        self.itermax = 5000
        
        self.fl = 2.5
        self.AP = 0.05
        
    def init__params(self):
        params = {'epochs':self.ep,
                  'filter1':self.f1,
                  'kernel1':self.k1,
                  'activation1':self.a1,
                  'filter2':self.f2,
                  'kernel2':self.k2,
                  'activation2':self.a2,
                  'pool_size':(2,2),
                  'dropout1':self.d1,
                  'unit1':self.u1,
                  'activation3':self.a3,
                  'dropout2':self.d2,
                  'activation4':self.a4,
                  'loss':self.lf,
                  'optimizer':self.op}
        return params
        
def init_net(p):
    return [Net() for _ in range(p)]

def fitness(n,n_c,i_shape,x,y,b,x_test,y_test):
    for cnt,i in enumerate(n):
        p = i.init__params()
        ep = p['epochs']
        f1 = p['filter1']        
        f2 = p['filter2']
        k1 = p['kernel1']        
        k2 = p['kernel2']
        d1 = p['dropout1']
        d2 = p['dropout2']
        ps = p['pool_size']
        u1 = p['unit1']        
        a1 = p['activation1']
        a2 = p['activation2']        
        a3 = p['activation3']
        a4 = p['activation4']
        lf = p['loss']
        op = p['optimizer']
        
        try:
            m = net_model(ep=ep,
                          f1=f1,
                          f2=f2,
                          k1=k1,
                          k2=k2,
                          a1=a1,
                          a2=a2,
                          a3=a3,
                          a4=a4,
                          d1=d1,
                          d2=d2,
                          u1=u1,
                          ps=ps,
                          op=op,
                          lf=lf,
                          n_c=n_c,
                          i_shape=i_shape,
                          x=x,
                          y=y,
                          b=b,
                          x_test=x_test,
                          y_test=y_test)
            
            s = m.evaluate(x=x_test,y=y_test,verbose=0)
            i.ac = s[1]
            print('Accuracy:{}'.format(i.ac*100))
        except Exception as e:
            print(e)
    return n            
        
def net_model(ep,f1,f2,k1,k2,a1,a2,a3,a4,d1,d2,u1,ps,op,lf,n_c,i_shape,x,y,b,x_test,y_test):
    model = Sequential()
    model.add(layer = Conv2D(filters = f1,kernel_size = k1,activation = a1,input_shape = i_shape))       
    model.add(layer = Conv2D(filters = f2,kernel_size = k2,activation = a2))
    model.add(layer=MaxPooling2D(pool_size=ps))
    model.add(layer=Dropout(rate=d1))
    model.add(layer=Flatten())
    model.add(layer=Dense(units=u1, activation=a3))
    model.add(layer=Dropout(rate=d2))
    model.add(layer=Dense(units=n_c, activation=a4))
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    model.compile(optimizer=op, loss=lf, metrics=['accuracy'])
    model.fit(x=x, y=y, batch_size=b, epochs=ep, verbose=0, validation_data=(x_test, y_test))
    return model
    
def selection(n): 
    n = sorted(n,key=lambda j:j.ac,reverse=True)
    n = n[:int(len(n))]
    return n

def crossover(n):
    offspring = []
    p1 = choice(n)
    #print("p1:",p1)
    p2 = choice(n)
    #print("p2:",p2)
    c1 = Net()
    c2 = Net()
    c1.ep = int(p2.ep) + 2
    c2.ep = int(p1.ep) + 2
    offspring.append(c1)
    offspring.append(c2)
    n.extend(offspring)
    return n

def mutate(n):
    for i in n:
        if uniform(0,1) <=0.1:
            i.ep +=randint(0,5)
            i.u1 +=randint(0,5)
    return n

if __name__ == '__main__':
    P = 10
    G = 100
    B = 128
    C = 10
    T = 0.994
    N = init_net(p=P)
    x_tr,x_te,y_tr,y_te,i_sh = mnist(show_examples=True)
    accuracy_list = []
    for g in range(G):
        print('Generator{}'.format(g + 1))
        N = fitness(n=N,
                    n_c=C,
                    i_shape=i_sh,
                    x=x_tr,
                    y=y_tr,
                    b=B,
                    x_test=x_te,
                    y_test=y_te)
        N = selection(n=N)
        N = crossover(n=N)
        N = mutate(n=N)
        
        for q in N:
            accuracy_list.append(q.ac*100)
            if q.ac > T:
                print('Threshold satisfied')
                print(q.init__params())
                print('Best accuracy:{}'.format(q.ac*100))
                exit()
                
        print("The best accuracy so far {}%".format(max(accuracy_list)))
                
            

    