# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 20:12:31 2018

@author: shadyrecords
"""

import tensorflow as tf

import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

sess = tf.InteractiveSession()
x = tf.placeholder("float",shape=[None,784])
y_ = tf.placeholder("float",shape=[None,10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.initialize_all_variables())

y = tf.nn.softmax(tf.matmul(x,W)+b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
for i in range(10000):
    batch = mnist.train.next_batch(50)
    train_step.run( feed_dict={x:batch[0],y_:batch[1]})

correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

print(accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels}))

