# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 11:40:35 2018

@author: zhaoyuzhi
"""

from PIL import Image
import layer_definition as nl   #build a new layer and get ImageToMatrix function
import tensorflow as tf
import xlrd
import numpy as np
import pandas as pd

IMAGESAVEURL_F1_CelebA = "G:\\CelebA\\Img\\img_align_celeba_F1"
IMAGECASIA_test = "C:/Users/zhaoyuzhi/Desktop/CASIA_test"
landmarks_excel = xlrd.open_workbook('CelebA_align_landmarks.xlsx')
landmarks_table = landmarks_excel.sheet_by_index(0)
CASIA_test_excel = xlrd.open_workbook('list.xlsx')
CASIA_test_table = CASIA_test_excel.sheet_by_index(0)
x_data = np.zeros([202599,39,39], dtype = np.float32)                #input imagematrix_data
y_data = np.ones([202599,10], dtype = np.float32)                    #correct output landmarks_data
CASIA_test = np.zeros([4442,39,39], dtype = np.float32)             #dataset to be processed

newlandmarks = np.zeros(10, dtype = np.float32)

## handle train data
for i in range(202599):                                               #train data part 1
    #get 39*39 numpy matrix of a single image
    imagename = landmarks_table.cell(i+1,0).value
    imagematrix = nl.ImageToMatrix(IMAGESAVEURL_F1_CelebA + '\\' + imagename)
    #extract image size and rawlandmarks data for normalized newlandmarks
    width = 89
    height = 89
    rawlandmarks = landmarks_table.row_slice(i+1, start_colx=1, end_colx=11)
    #get ten normalized newlandmarks(coordinates of LE,RE,N,LM,RM)
    for j in range(0,9,2):
        newlandmarks[j] = (rawlandmarks[j].value - 44) / width * 39
    for k in range(1,10,2):
        newlandmarks[k] = (rawlandmarks[k].value - 90) / height * 39
    #one dimension which represents one grey picture, set the first dimension as index
    x_data[i,:,:] = imagematrix
    y_data[i,:] = newlandmarks

# read actual test data
for i in range(4442):
    #get 39*39 numpy matrix of a single image
    imagename = CASIA_test_table.cell(i,0).value
    img = Image.open(IMAGECASIA_test + '/' + imagename,'r')
    img = img.crop([22,22,122,122])
    img_resize = img.resize((39,39), Image.ANTIALIAS)
    imgdata = img_resize.getdata()
    npdata = np.matrix(imgdata, dtype= np.float32) / 255.0
    newdata = np.reshape(npdata, (39,39))
    CASIA_test[i,:,:] = newdata

## F1
x = tf.placeholder(tf.float32, shape=[None,39,39], name='x')        #input imagematrix_data to be fed
y = tf.placeholder(tf.float32, shape=[None,10], name='y')           #correct output to be fed
keep_prob = tf.placeholder(tf.float32, name='keep_prob')            #keep_prob parameter to be fed

x_image = tf.reshape(x, [-1,39,39,1])

## convolutional layer 1, kernel 4*4, insize 1, outsize 20
W_conv1 = nl.weight_variable([4,4,1,20])
b_conv1 = nl.bias_variable([20])
h_conv1 = nl.conv_layer(x_image, W_conv1) + b_conv1                 #outsize = batch*36*36*20
a_conv1 = tf.nn.relu(h_conv1)                                       #outsize = batch*36*36*20

## max pooling layer 1
h_pool1 = nl.max_pool_22_layer(a_conv1)                             #outsize = batch*18*18*20
a_pool1 = tf.nn.relu(h_pool1)                                       #outsize = batch*18*18*20

## convolutional layer 2, kernel 3*3, insize 20, outsize 40
W_conv2 = nl.weight_variable([3,3,20,40])
b_conv2 = nl.bias_variable([40])
h_conv2 = nl.conv_layer(a_pool1, W_conv2) + b_conv2                 #outsize = batch*16*16*40
a_conv2 = tf.nn.relu(h_conv2)                                       #outsize = batch*16*16*40

## max pooling layer 2
h_pool2 = nl.max_pool_22_layer(a_conv2)                             #outsize = batch*8*8*40
a_pool2 = tf.nn.relu(h_pool2)                                       #outsize = batch*8*8*40

## convolutional layer 3, kernel 3*3, insize 40, outsize 60
W_conv3 = nl.weight_variable([3,3,40,60])
b_conv3 = nl.bias_variable([60])
h_conv3 = nl.conv_layer(a_pool2, W_conv3) + b_conv3                 #outsize = batch*6*6*60
a_conv3 = tf.nn.relu(h_conv3)                                       #outsize = batch*6*6*60

## max pooling layer 3
h_pool3 = nl.max_pool_22_layer(a_conv3)                             #outsize = batch*3*3*60
a_pool3 = tf.nn.relu(h_pool3)                                       #outsize = batch*3*3*60

## convolutional layer 4, kernel 2*2, insize 60, outsize 80
W_conv4 = nl.weight_variable([2,2,60,80])
b_conv4 = nl.bias_variable([80])
h_conv4 = nl.conv_layer(a_pool3, W_conv4) + b_conv4                 #outsize = batch*2*2*80
a_conv4 = tf.nn.relu(h_conv4)                                       #outsize = batch*2*2*80

## flatten layer
x_flat = tf.reshape(a_conv4, [-1,320])                              #outsize = batch*320

## fully connected layer 1
W_fc1 = nl.weight_variable([320,120])
b_fc1 = nl.bias_variable([120])
h_fc1 = tf.matmul(x_flat, W_fc1) + b_fc1                            #outsize = batch*120
a_fc1 = tf.nn.relu(h_fc1)                                           #outsize = batch*120
a_fc1_dropout = tf.nn.dropout(a_fc1, keep_prob)                     #dropout layer 1

## fully connected layer 2
W_fc2 = nl.weight_variable([120,10])
b_fc2 = nl.bias_variable([10])
h_fc2 = tf.matmul(a_fc1_dropout, W_fc2) + b_fc2                     #outsize = batch*10
a_fc2 = tf.nn.relu(h_fc2)                                           #outsize = batch*10

#regularization and loss function
original_cost = tf.reduce_mean(tf.pow(y - a_fc2, 2))
tv = tf.trainable_variables()   #L2 regularization
regularization_cost = 1 / 32 * tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])   #1 / 32 is hyperparameter
cost = original_cost + regularization_cost
Optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)
init = tf.global_variables_initializer()
cache_F1 = np.zeros([4442,10], dtype = np.float32)

with tf.Session() as sess:
    sess.run(init)
    for i in range(100):                                            #number of iterations:1266200, 20259200 images
        for m in range(12662):                                      #training process using training data 202592 images
            train_xbatch = x_data[(m*16):(m*16+16),:,:]             #train 16 data every batch, not including m*16+16
            train_ybatch = y_data[(m*16):(m*16+16),:]               #train 16 data every batch, not including m*16+16
            sess.run(Optimizer, feed_dict = {x:train_xbatch, y:train_ybatch, keep_prob:0.5})

    for k in range(2221):
        test_xbatch = CASIA_test[(k*2):(k*2+2),:,:]                 #train 2 data every batch, not including m*2+2
        cache_F1[(k*2):(k*2+2),:] = a_fc2.eval(feed_dict = {x:test_xbatch, keep_prob:1})
       
print(cache_F1)

## save the predicted keypoints to excel
cache_F1_df = pd.DataFrame(cache_F1)
writer = pd.ExcelWriter('F1_CelebA.xlsx')
cache_F1_df.to_excel(writer,'sheet1')
writer.save()
