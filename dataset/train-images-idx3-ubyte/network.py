# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 20:44:05 2018

@author: shadyrecords

http://neuralnetworksanddeeplearning.com/chap1.html
"""
import numpy as np
import LoadMnist as lm
from sklearn import model_selection
import random 



class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]
        


    
    
    def feedforward(self,a):
        '''
        return the output of the network if 'a' is input
        '''
       
        for b,w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w,a)+b)
        
        return a
    
    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                        training_data[k:k+mini_batch_size]
                        for k in range(0,n,mini_batch_size)
                        ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print("Epoch {0}:{1}/{2}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {0} complete".format(j))
    
    def update_mini_batch(self,mini_batch,eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            
            delta_nabla_b,delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        
        self.weights = [w-(eta/len(mini_batch)) * nw
                        for w, nw in zip(self.weights,nabla_w)
                        ]
        self.biases = [b-(eta/len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]
    
    def backprop(self,x,y):
        """Return a tuple "(nabla_b, nabla_w)" representing the
        gradient for the cost function C_x.  "nabla_b" and
        "nabla_w" are layer-by-layer lists of numpy arrays, similar
        to "self.biases" and "self.weights"."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        #x= x.reshape(-1,1)
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            #print(w.shape)
            #print(b.shape)
            #print(activation.shape)
            z = np.dot(w, activation)+b
            #print(z.shape)
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.

            
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)    

   
    def evaluate(self, test_data):
        
 
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
        #print('---')
        print(test_results[4])
        
        return sum(int(x == y) for (x, y) in test_results)


    def cost_derivative(self,output_activations,y):
         return (output_activations - y)
    

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

if __name__ == '__main__':
    x_train,y_train = lm.load_mnist('')
    
    x_tr,x_te,y_tr,y_te = model_selection.train_test_split(x_train,y_train,train_size = 0.1,test_size=0.01)
    #print(x_tr.shape)
    #print(y_tr.shape)
    #print(x_te.shape)
    #print(y_te.shape)
    def vectorized_result(j):
        """Return a 10-dimensional unit vector with a 1.0 in the jth
        position and zeroes elsewhere.  This is used to convert a digit
        (0...9) into a corresponding desired output from the neural
        network."""
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e
    x_tr = [np.reshape(i,(784,1) )for i in x_tr]
    x_te = [np.reshape(i,(784,1) )for i in x_te]
    y_tr = [vectorized_result(y) for y in y_tr]
    y_te = [vectorized_result(y) for y in y_te]
    
    training_data = list(zip(x_tr,y_tr))
    #print(training_data[0])
    validation_data = list(zip(x_te,y_te))
    net = Network([784,30,10])
    net.SGD(training_data, 100, 10, 3, test_data=validation_data)
    #for x,y in validation_data:
        
