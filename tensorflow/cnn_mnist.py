# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 22:09:04 2018

@author: shadyrecords
"""

"""
convolutional layer 1: applies 32 5*5 filters(extracting 5*5-
pixel subregions),with Relu activation function

pooling layer 1: performs max pooling with a 2x2 filters
and stride of 2

convolutional layer 2: Applies 64 5x5 filters,with Relu 
acitivation function  

pooling layer 2: 2x2 filters and stride of 2

dense layer 1： 1024 neurons with dropout regularization
rate of 0.4

dense layer 2： 10 neurons,one for each digit target class(0-9)
                                             
"""
import numpy as np
import tensorflow as tf

def cnn_model_fn(features,labels,mode):
    """
    Model function for CNN.
    """
    # input layer
    #  shape of input layer 
    #[channels, batch_size, image_width, image_height]
    input_layer = tf.reshape(features["x"],[-1,28,28,1])
    
    # convolutional layer 1
    # return tensor of shape [batch_size,28,28,32]
    conv1 = tf.layers.conv2d(
            inputs = input_layer,
            filters = 32,
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu)
    
    #pooling layer 
    #return a shape of [batch_size,14,14,32]
    #the 2*2 filter reduces width and height by %50 each
    
    pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2,2],
            strides=2)
    
    #convolutional layer2
    #the pooling layer return a shape of 
    #[batch_size,7,7,64]
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(
            inputs = conv2,
            pool_size = [2,2],
            strides=2)
    
    # dense layer
    #flat first 
    pool2_flat = tf.reshape(pool2,[-1,7*7*64])
    dense = tf.layer.dense(
            inputs = pool2_flat,
            units = 1024, # the number of neurons 
            activaton = tf.nn.relu)
    dropout = tf.layers.dropout(
            inputs=dense,
            rate=0.4, # 40% of the elements will be randomly dropped during training
            # the training argument takes a boolean specifying
            # whether or not the model is currently being run in training mode
            # because dropout will only be performed if training is True
            
            training=mode==tf.estimator.ModeKeys.TRAIN)
    
    # logits layer
    # will return the raw values for our predictions
    # the final output tensor shape will be [batch_size,10]
    logits = tf.layers.dense(inputs=dropout,units=10)
    
    predictions = {
            #gernerate predictions (for predict and eval mode)
            # get the highest raw value as the precicted class
            # the axis argument specifies the axis of the input 
            # tensor along which to find the greatest value.  
            "classes":tf.argmax(input=logits,axis=1),
            
            #add "softmax_tensor" to the graph. it is used for predict
            #and bt the logging_hook
            "probabilities": tf.nn.softmax(logits,name="softmax_tensor")
            }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions)
        
    # calculate loss (for both TRAIN and EVAL modes
    # for both training and evaluaiton,we need to define a 
    # loss function 
    
    # why onehot?
    # the tf.one_hot has two required arguments
    # indices 
    # depth is 10 because we have 10 possible target classes
    
    onehot_labels = tf.one_hot(indices=tf.cast(labels,tf.int32),depth=10)
    
    # tf.losses.softmax_cross_entropy return a scalar tensor
    loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels,logits=logits)
    
    # configure the training op
    # after we have defined loss and structure of our CNN
    # now we configure our model to optimize this loss value
    # during training 
    
    # using a learning rate of 0.001 and sgd as the 
    # optimization algorithm
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op = train_op)
    # add evalution metrics'
    eval_metric_ops = {
            "accuracy" : tf.metrics.accuracy(
                    labels=labels,predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
            mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    mnsit = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.label,dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels,dtype=np.int32)
    tensors_to_log = {"probabilities":"softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log,every_n_iter=50)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x = {"x":train_data},
            y = train_labels,
            batch_size=100,
            num_epochs=None,
            shuffle=True)
    mnist_classifier.train(
            input_fn=train_input_fn,
            steps=20000,
            hooks=[logging_hook])
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x":eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    
    

mnist_classifier = tf.estimator.Estimator(
        model_fn = cnn_model_fn,model_dir='.')

if __name__ == "__main__":
    tf.app.run()
    