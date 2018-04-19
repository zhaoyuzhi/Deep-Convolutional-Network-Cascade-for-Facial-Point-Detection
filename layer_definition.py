# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 10:58:01 2018

@author: zhaoyuzhi
"""

import tensorflow as tf
from PIL import Image
import numpy as np

def weight_variable(shape):
    #Truncated normal distribution function
    #shape is kernel size, insize and outsize
	initial = tf.truncated_normal(shape, stddev = 0.1, dtype = tf.float32)
	return tf.Variable(initial)

def bias_variable(shape):
    #initialize bias as a constant number
    initial = tf.constant(0.1, shape = shape, dtype = tf.float32)
    return tf.Variable(initial)

def conv_layer(x, W):
    #stride = [1,x_movement,y_movement,1]
    #must have strides[0] = strides[3] = 1 
    #because they represent number of images and channels respectively
    #according to the paper, the padding is valid
    #the input kernel size 'W' should be predifined before using conv_layer
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding='VALID')

def max_pool_22_layer(x):
    #stride = [1,x_movement,y_movement,1]
    #must have strides[0] = strides[3] = 1
    #ksize(kernel size) = [1,length,height,1]
    #according to the paper, we use 2x2 max-pooling
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding='VALID')

def flatten_layer(x):
    #x_shape = [num_images, img_height, img_width, num_channels]
    x_shape = x.get_shape()
    #change to [num_images, num_features]
    #num_features = x_shape[1] * x_shape[2] * x_shape[3]
    #get reshaped x whose dimension is [num_images, img_height * img_width * num_channels]
    x_flat = tf.reshape(x, [x_shape[0], x_shape[1] * x_shape[2] * x_shape[3]])
    return x_flat

def fc_layer(x, W, b):
    #the input weights 'W' and bias 'b' should be predifined before using fc_layer
    return (tf.matmul(x, W) + b)

def ImageToMatrix(filename):
    #get numpy matrix of a image
    image = Image.open(filename)
    #image = image.resize((28,28), Image.ANTIALIAS)   #this is a test for author:zhaoyuzhi
    width, height = image.size
    imgdata = image.getdata()
    npdata = np.matrix(imgdata, dtype='float32') / 255.0
    newdata = np.reshape(npdata, (height, width))
    return newdata

def Img15ToMatrix(img):
    img_resize = img.resize((15,15), Image.ANTIALIAS)
    img_resize_data = img_resize.getdata()
    npdata = np.matrix(img_resize_data, dtype='float32') / 255.0
    newdata = np.reshape(npdata, (15,15))
    return newdata
