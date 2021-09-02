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
from keras.models import Sequential
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

class Net(object):
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
        
    def init_params(self):
        params = {'epoch':self.ep,
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
    print(p)
    return [Net() for _ in range(p)]

def net_model(ep,f1,f2,k1,k2,a1,a2,a3,a4,d1,d2,u1,ps,op,lf,n_c,i_shape,x,y,b,x_test,y_test):
    model = Sequential()
    model.add(layer=Conv2D(filters = f1,kernel_size = k1,activation = a1,input_shape = i_shape))       
    model.add(layer=Conv2D(filters = f2,kernel_size = k2,activation = a2))
    model.add(layer=MaxPooling2D(pool_size = ps))
    model.add(layer=Dropout(rate = d1))
    model.add(layer=Flatten())
    model.add(layer=Dense(units = u1, activation = a3))
    model.add(layer=Dropout(rate = d2))
    model.add(layer=Dense(units = n_c, activation = a4))
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    model.compile(optimizer=op, loss=lf, metrics=['accuracy'])
    model.fit(x=x, y=y, batch_size=b, epochs=ep, verbose=0, validation_data=(x_test, y_test))
    return model

x = init_net(10) 

print(x[0].init_params())

    
print(x[0].init_params())
x_tr, x_te, y_tr, y_te, i_sh = mnist(show_examples=True)
n_c = 10
accary_list = []
for count , i in enumerate(x):
    print('count:',count)
    print('i:',i.init_params())
    z = i.init_params()
    ep = z['epoch']
    f1 = z['filter1']
    k1 = z['kernel1']
    a1 = z['activation1']
    f2 = z['filter2']
    k2 = z['kernel2']
    a2 = z['activation2']
    ps = z['pool_size']
    d1 = z['dropout1']
    u1 = z['unit1']
    a3 = z['activation3']
    d2 = z['dropout2']
    a4 = z['activation4']
    lf = z['loss']
    op = z['optimizer']
    
    try:
        m = net_model(ep = ep,
                      f1 = f1,
                      f2 = f2,
                      k1 = k1,
                      k2 = k2,
                      a1 = a1,
                      a2 = a2,
                      a3 = a3,
                      a4 = a4,
                      d1 = d1,
                      d2 = d2,
                      u1 = u1,
                      ps = ps,
                      op = op,
                      lf = lf,
                      n_c = n_c, 
                      i_shape=i_sh,
                      x = x_tr,
                      y = y_tr,
                      b = 128,
                      x_test = x_te,
                      y_test = y_te)
        s = m.evaluate(x=x_te,y=y_te,verbose=0)
        i.ac = s[1]
        accary_list.append(i.ac)
        print("accary_list:",accary_list)
        accary_list_np = np.array(accary_list)
        print("accary_list_np:",accary_list_np)
        print('Accuracy:{}'.format(i.ac*100))
    except Exception as e:
        print(e)

        

