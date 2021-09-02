# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:26:06 2020

@author: user
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from abc import abstractmethod
class GA:
    def __init__(self,x_train,x_test,y_train,y_test,pop_size,
                 r_mutation,p_crossover,p_mutation,max_iter,min_fitness,elite_num,
                 mating_pool_size,batch_size,dataset='cifar10'):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.pop_size = pop_size
        self.r_mutation = r_mutation
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.max_iter = max_iter
        self.min_fitness = min_fitness
        self.elite_num = elite_num
        self.mating_pool_size = mating_pool_size
        self.batch_size = batch_size
        self.dataset = dataset
        
        self.chroms = []
        self.evaluation_history = []
        self.stddev = 0.5
        self.loss_func = 'categorical_crossentropy'
        self.metrics = ['accuracy']
        
    def cur_iter(self):
        print(type(len(self.evaluation_history)))
        return len(self.evaluation_history)
    
    def shuffle_batch(self):
        series = list(range(len(self.x_train)))
        np.random.shuffle(series)
        return series
    
    def initialization(self):
        for i in range(self.pop_size):
            model = Sequential()
            if self.dataset == 'cifar10':
                model.add(Conv2D(6,(1,1),activation = 'relu',use_bias = False,input_shape=(32, 32, 3)))
                model.add(AveragePooling2D((2,2)))
                model.add(Conv2D(16,(1,1),activation = 'relu',use_bias = False))
                model.add(AveragePooling2D((2,2)))
                model.add(Conv2D(120,(1,1),activation = 'relu',use_bias = False))
                model.add(AveragePooling2D((2,2)))
                model.add(Flatten())
                model.add(Dense(120,activation = 'relu',use_bias = False))
                model.add(Dense(84,activation = 'relu',use_bias = False))
                model.add(Dense(10,activation = 'relu',use_bias = False))
            elif self.dataset == 'mnist':
                model.add(Conv2D(40,(1, 1),activation = 'relu',use_bias= False,input_shape=(28, 28, 1)))
                model.add(AveragePooling2D((2, 2)))
                model.add(Conv2D(40,(1, 1),activation = 'relu',use_bias = False))
                model.add(AveragePooling2D((2, 2)))
                model.add(Conv2D(5,(1, 1),activation = 'relu',use_bias = False))
                model.add(AveragePooling2D((2, 2)))
                model.add(Conv2D(1,(1, 1),activation = 'relu',use_bias = False))
                model.add(Flatten())
                model.add(Dense(40,activation = 'relu',use_bias = False))
                model.add(Dense(10,activation = 'softmax', use_bias = False))
            else:
                raise Exception('Unknown dataset.')
            self.chroms.append(model)
            
        print('{} network initialization({}) finished.'.format(self.dataset, self.pop_size))
    
    def evaluation(self,x_te,y_te,is_batch = True):
        cur_evaluation = []
        for i in range(self.pop_size):
            model = self.chroms[i]
            model.compile(loss = self.loss_func,metrics = self.metrics, optimizer = 'adam')
            train_loss , train_acc = model.evaluate(x_te,y_te,verbose = 0)
            if not is_batch:
                test_loss , test_acc = model.evaluate(self.x_test,self.y_test,verbose = 0)
                cur_evaluation.append({
                    'pop': i,
                    'train_loss': round(train_loss, 4),
                    'train_acc': round(train_acc, 4),
                    'test_loss': round(test_loss, 4),
                    'test_acc': round(test_acc, 4),
                })
            else:
                cur_evaluation.append({
                    'pop': i,
                    'train_loss': round(train_loss, 4),
                    'train_acc': round(train_acc, 4),
                })
        
        best_fit = sorted(cur_evaluation, key=lambda x: x['train_acc'])[-1]
        self.evaluation_history.append({
            'iter':self.cur_iter() + 1,
            'best_fit':best_fit,
            'avg_fitness':np.mean([e['train_acc'] for e in cur_evaluation]).round(4),
            'evaluation': cur_evaluation,
            })
        print('\niter:{}'.format(self.evaluation_history[-1]['iter']))
        print('Best_fit:{},avg_fitness:{:.4f}'.format(self.evaluation_history[-1]['best_fit'],
                                                      self.evaluation_history[-1]['avg_fitness']))
        
    def roulette_wheel_selection(self):
        sorted_evaluation = sorted(self.evaluation_history[-1]['evaluation'],key = lambda x:x['train_acc'])
        cum_acc = np.array([e['train_acc'] for e in sorted_evaluation]).cumsum()
        extra_evaluation = [{'pop': e['pop'], 'train_acc': e['train_acc'], 'cum_acc': acc}
                            for e, acc in zip(sorted_evaluation, cum_acc)]
        rand = np.random.rand() * cum_acc[-1]
        for e in extra_evaluation:
            if rand < e['cum_acc']:
                return e['pop']
            return extra_evaluation[-1]['pop']
        
    @abstractmethod
    def run(self):
        raise NotImplementedError('Run not implemented.')

    @abstractmethod
    def selection(self):
        raise NotImplementedError('Selection not implemented.')

    @abstractmethod
    def crossover(self, _selected_pop):
        raise NotImplementedError('Crossover not implemented.')

    @abstractmethod
    def mutation(self, _selected_pop):
        raise NotImplementedError('Mutation not implemented.')

    @abstractmethod
    def replacement(self, _child):
        raise NotImplementedError('Replacement not implemented.')
        
    

    
    