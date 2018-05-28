import tensorflow as tf
import pickle
import utils
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib import rnn
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import numpy as np

class sp_interactive_layer:
    def __init__(self,input_tensor,channels=2,filter_num=8,kernel_size=(3,3),pool_size=(2,2),strides=(2,2),con_times=1):
        input_dims=input_tensor.shape[2]
        self.W=tf.get_variable('sp_w',shape=(input_dims,channels,input_dims),
                               initializer=xavier_initializer(), dtype=tf.float32)
        # input_tensor shape is (batchsize,m,n),we calculate an image with shape (batch_size,m,m,channels)
        tmp = tf.tensordot(input_tensor, self.W, axes=[[2], [0]])#batchsize,m,channels,n
        mv_t = tf.transpose(input_tensor, perm=[0, 2, 1])#batchsize,n,m
        mv_t = tf.stack([mv_t] * channels, axis=1)#batchsize,channels,n,m
        matching_image = tf.matmul(tf.transpose(tmp, perm=[0, 2, 1, 3]), mv_t)#batchsize,channels,n,n
        matching_image = tf.transpose(matching_image, perm=[0, 2, 3, 1])#batchsize,n,n,channels
        for i in range(0,con_times):
            conv_layer = tf.layers.conv2d(matching_image, filters=filter_num, kernel_size=kernel_size, padding='VALID',
                                      kernel_initializer=tf.contrib.keras.initializers.he_normal(),
                                      activation=tf.nn.leaky_relu, name='conv_'+str(i))  # TODO: check other params
            matching_image = tf.layers.max_pooling2d(conv_layer, pool_size=pool_size, strides=strides,
                                                padding='VALID', name='max_pooling_'+str(i))  # TODO: check other params
        self.final_matching_vector=tf.contrib.layers.flatten(matching_image)
    def call(self):
        return self.final_matching_vector