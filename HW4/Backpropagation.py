#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv("data/data.txt", names = ['x', 'y', 'label'])

# data


# In[3]:


import random


def init_network(in_num = 2, hid_num = 1, hid_layer = 2, out_num = 2):
    net = []
    
    #init first hidden layer
    layer = []
    for num in range(hid_num):
        neuron = {'w': [random.random() for j in range(in_num+1)]}
        layer.append(neuron)
    net.append(layer)
    
    #init remain hidden layer
    for i in range(hid_layer-1):
        layer = []
        for num in range(hid_num):
            neuron = {'w': [random.random() for j in range(hid_num+1)]}
            layer.append(neuron)
        net.append(layer)
    
    #init output layer
    layer = []
    for num in range(out_num):
        neuron = {'w': [random.random() for j in range(hid_num+1)]}
        layer.append(neuron)
    net.append(layer)
    
    return net


# In[4]:


import math
def sigmoid(x):
    try:
        ans = 1.0/(1.0+math.exp(-x))
    except OverflowError:
        ans = 0.0000001
    return ans
def sigmoid_deriv(x):
    return x * (1.0 - x)


# In[5]:


def weight_sum(weights, inputs):
    bias = weights[-1]
    sum_w = bias
    
    for i in range(len(weights)-1):
        sum_w += weights[i]*inputs[i]
    return sigmoid(sum_w)
    
def forward_prop(network, inputs):
    
    for layer in network:
        neuron_sum = []
        for neuron in layer:
            neuron['out'] = weight_sum(neuron['w'], inputs)
            neuron_sum.append(neuron['out'])
            
        inputs = neuron_sum
        
    return inputs

# network = init_network()
# inp = [1, 0]
# ans = forward_prop(network, inp)
# ans


# In[6]:


def cross_entropy_derv(x, y):
    return x-y
            
def backward_prop(network, label):
    for idx in range(len(network)-1, -1, -1):
        #output layer
        if idx==len(network)-1:
            j = 0
            for neuron in network[idx]:
                n_out = neuron['out']
                neuron['err'] = cross_entropy_derv(label[j], n_out)*sigmoid_deriv(n_out)
                j+=1
    
        else:
            for i in range(len(network[idx])):
                neuron = network[idx][i]
                neuron['err'] = 0.0
                
                for prev_neuron in network[idx+1]:
                    neuron['err'] += prev_neuron['w'][i] * prev_neuron['err']
            
                neuron['err'] *= sigmoid_deriv(neuron['out'])


# In[14]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
    

def train(network, dataset, epochs=20):
    time = 0
    e = 1e-9
    for epoch in range(epochs):
        err = 0
        for x, y, label in zip(dataset['x'], dataset['y'], dataset['label']):
#         for x, y, label in dataset:
            outcome = forward_prop(network, [x, y])  
            if label==1:
                labels = [0, 1]
            elif label==0:
                labels = [1, 0]
            
            err += ((labels[0]-outcome[0])**2 + (labels[1]-outcome[1])**2)
#             err += (label - outcome[0])**2
            backward_prop(network, labels)
#             backward_prop(network, [label])

            # updates weights
            
            learn = 0.5 
            time+=1
            # learn = 0.010
            for idx in range(len(network)):
                layer = network[idx]
                if idx==0:
                    inputs = [x, y]
                else:
                    inputs = [neuron['out'] for neuron in network[idx-1]]
                    
                for neuron in layer:
                    nums = len(neuron['w'])-1
                    # weights
                    for num in range(len(inputs)):
                        if 'lr' not in neuron:
                            neuron['lr'] = 0.0
                        
                        #Do Adagram    
                        neuron['lr'] = neuron['lr'] + (neuron['err'] * inputs[num])**2
                        
                        neuron['w'][num] += (learn / (e+math.sqrt(neuron['lr'])) * (neuron['err'] * inputs[num]) )
                    # bias
                    neuron['w'][-1] += (learn / (e+math.sqrt(neuron['lr'])) * neuron['err'])
                
        if ((epoch+1) %500) == 0:
            print('>epoch= {0}, learning= {1}, error= {2}'.format(epoch+1, "Adagram", err))
            print_acc(network, dataset)
def print_acc(network, dataset):
    correct = 0
    for x, y, label in zip(dataset['x'], dataset['y'], dataset['label']):
        outcome = forward_prop(network, [x, y])
        if x > y and outcome[0] > outcome[1]:
            correct += 1
        elif x < y and outcome[0] < outcome[1]:
            correct += 1
    print('Accuracy: {0}'.format(correct / 100))

def plot_result(network, dataset):
    cluster1 = {'x':[], 'y':[]}
    cluster2 = {'x':[], 'y':[]}
    ground1 = {'x' : [], 'y':[]}
    ground2 = {'x' : [], 'y':[]}
    correct = 0
    for x, y, label in zip(dataset['x'], dataset['y'], dataset['label']):
        outcome = forward_prop(network, [x, y])
        if (outcome[0] > outcome[1]):
#         if(outcome[0] < 0.5):
            if x > y:
                correct += 1
            cluster1['x'].append(x)
            cluster1['y'].append(y)
        else:
            if x < y:
                correct += 1
            cluster2['x'].append(x)
            cluster2['y'].append(y)
        
        if x > y:
            ground1['x'].append(x)
            ground1['y'].append(y)
        elif x < y:
            ground2['x'].append(x)
            ground2['y'].append(y)
#     fig, (ax1, ax2) = plt.subplots(2)
    
    plt.title('Prediction')
    plt.plot(cluster1['x'], cluster1['y'],'ro', label='x < y')
    plt.plot(cluster2['x'], cluster2['y'], 'bD', label='x > y')
    plt.plot([0, 100], [0, 100])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    
    plt.title('Ground Truth')
    plt.plot(ground1['x'], ground1['y'], 'ro', label='x < y')
    plt.plot(ground2['x'], ground2['y'], 'bD', label='x > y')
    plt.plot([0, 100], [0, 100])
    plt.xlabel('x')
    plt.ylabel('y')

    plt.legend()
    plt.show()
    print('Accuracy: {0}'.format(correct / 100))
    
def print_result(network, dataset):
    idx = 0
    for x, y, label in zip(dataset['x'], dataset['y'], dataset['label']):
        idx += 1
        outcome = forward_prop(network, [x, y])
        print('Point {4}: x = {0}, y = {1}, label = {2}, outcome[0, 1] = {3}'.format(x, y, label, outcome, idx))
        
def print_NN(network):
    print('input layer: {0} neuron'.format(2))
    for i in range(len(network)):
        print('weights and bias between layers= ')
        for neuron in network[i]:
            print(neuron['w'])
        print('')

        if i==len(network)-1:
            print('output layer: {0} neuron'.format(len(network[i])))
        else:
            print('hidden layer {0}: {1} neuron'.format(i+1, len(network[i])))

if __name__ == '__main__':
             
    network = init_network(2, 2, 1, 2)
    train(network, data, 10000)

    plot_result(network, data)
    print_result(network, data)
    print_NN(network)


# In[ ]:




