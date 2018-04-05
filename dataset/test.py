# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 17:14:41 2018

@author: shadyrecords
"""
import numpy as np
import mnist_loader as lm
from sklearn import model_selection
import random  
import network

training_data,validation_data,test_data = lm.load_data_wrapper()

import net3
net = net3.Network([784,30,10])
net.SGD(training_data, 100, 30,3.0,1.0,\
        evaluation_data=validation_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True,
        early_stopping_n=100)