#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: your_model.py
# Brown CSCI 1430 assignment
# Created by Aaron Gokaslan

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.tower import get_current_tower_context
import tensorflow as tf
import hyperparameters as hp

class YourModel(ModelDesc):

    def __init__(self):
        super(YourModel, self).__init__()
        self.use_bias = True

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, hp.img_size, hp.img_size, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs

        #####################################################################
        # TASK 1: Change architecture (to try to improve performance)
        # TASK 1: Add dropout regularization                                  
        kp = .5
        # Declare convolutional layers
        #
        # TensorPack: Convolutional layer
        # 10 filters (out_channel), 9x9 (kernel_shape), 
        # no padding, stride 1 (default settings)
        # with ReLU non-linear activation function.
        logits = Conv2D('conv1', image, 50, (3,3), padding='valid', stride=(1,1), nl=tf.nn.relu)
        logits = MaxPooling('pool1', logits, (3,3), stride=None, padding='valid')
        logits = tf.nn.dropout(logits, kp)

        logits = Conv2D('conv2', logits, 300, (3,3), padding='valid', stride=(1,1), nl=tf.nn.relu)
        logits = MaxPooling('pool2', logits, (3,3), stride=None, padding='valid')
        logits = tf.nn.dropout(logits, kp)

        # logits = FullyConnected('fc0',  logits, hp.category_num, nl=tf.nn.relu)

        logits = Conv2D('conv0', logits, 600, (3,3), padding='valid', stride=(1,1), nl=tf.nn.relu)
        # Chain layers together using reference 'logits'
        # 7x7 max pool, stride = none (defaults to same as shape), padding = valid
        # logits = Conv2D('conv1', logits, 20, (3,3), padding='valid', stride=(1,1), nl=tf.nn.relu)

        

        # logits = Conv2D('conv3', logits, 50, (3,3), padding='valid', stride=(1,1), nl=tf.nn.relu)
        # logits = MaxPooling('pool3', logits, (3,3), stride=None, padding='valid')
        
        #logits = Conv2D('conv4', logits, 20, (3,3), padding='valid', stride=(1,1), nl=tf.nn.relu)
        #logits = MaxPooling('pool4', logits, (3,3), stride=(1,1), padding='valid')
        #logits = FullyConnected('fc1', logits, 10, nl=tf.nn.relu)
        #logits = tf.nn.dropout(logits, kp)
        logits = FullyConnected('fc2', logits, hp.category_num, nl=tf.identity)
        #####################################################################

        # Add a loss function based on our network output (logits) and the ground truth labels
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = prediction_incorrect(logits, label)

        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))


        #####################################################################
        # TASK 1: If you like, you can add other kinds of regularization, 
        # e.g., weight penalization, or weight decay


        #####################################################################


        # Set costs and monitor them for TensorBoard
        add_moving_summary(cost)
        add_param_summary(('.*/kernel', ['histogram']))   # monitor W
        self.cost = tf.add_n([cost], name='cost')


    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', hp.learning_rate, summary=True)
        m = get_scalar_var('momentum', hp.momentum, summary=True)
        # Use momentum based gradient descent as our optimizer
        opt = tf.train.MomentumOptimizer(lr, m)
        # opt = tf.train.GradientDescentOptimizer(lr)
        
        return opt