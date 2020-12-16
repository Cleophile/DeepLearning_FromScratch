#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Hikari Software
# Y-Enterprise

import tensorflow as tf
from tensorflow import keras

import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Define a network with no hidden layers
# placeholder for input variables x
x = tf.placeholder(tf.float32, [None, 784])

# Variable for trainable variables
# zeros for initial value
w = tf.Variable(tf.zeros([784, 10]))

# bias variable
b = tf.Variable(tf.zeros([10]))

# linear combinition first, matrix 
y = tf.nn.softmax(tf.matmul(x,w) + b)

# placeholder for labels
label = tf.placeholder(tf.float32, [None, 10])

train = 

print(x_train.shape)
print(y_train.shape)
print(y_train)
