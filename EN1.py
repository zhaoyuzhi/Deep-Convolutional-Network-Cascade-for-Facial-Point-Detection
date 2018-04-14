# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 14:55:43 2018

@author: zhaoyuzhi
"""

from PIL import Image
import layer_definition as nl   #build a new layer and get ImageToMatrix function
import tensorflow as tf
import xlrd
import numpy as np

IMAGEURL = "C:\\Users\\zhaoyuzhi\\Desktop\\train"
IMAGESAVEURL_EN1_lfw = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_EN1"
IMAGESAVEURL_EN1_net = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_EN1"
IMAGESAVEURL_EN1_lfw_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_EN1_test"
IMAGESAVEURL_EN1_net_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_EN1_test"
train = xlrd.open_workbook('trainImageList.xlsx')
train_table = train.sheet_by_index(0)
test = xlrd.open_workbook('testImageList.xlsx')
test_table = test.sheet_by_index(0)
x_data = np.zeros([10000,31,39], dtype = np.float32)                #input imagematrix_data
y_data = np.ones([10000,6], dtype = np.float32)                     #correct output landmarks_data
x_test = np.zeros([3466,31,39], dtype = np.float32)
y_test = np.ones([3466,6], dtype = np.float32)

landmarks = np.zeros(6, dtype = np.float32)
newlandmarks = np.zeros(6, dtype = np.float32)

## handle train data
for i in range(4151):                                               #train data part 1
    #get 39*39 numpy matrix of a single image
    imagename = train_table.cell(i+1,0).value
    true_imagename = imagename[9:]
    imagematrix = nl.ImageToMatrix(IMAGESAVEURL_EN1_lfw + '\\' + true_imagename)
    #extract image size and rawlandmarks data for normalized newlandmarks
    img = Image.open(IMAGEURL + '\\' + imagename,'r')
    [width, height] = img.size
    rawlandmarks = train_table.row_slice(i+1, start_colx=5, end_colx=11)
    #get ten normalized newlandmarks(coordinates of LE,RE,N,LM,RM)
    for l in range(6):
        landmarks[l] = rawlandmarks[l].value
    for j in range(0,5,2):
        newlandmarks[j] = landmarks[j]/width * 39
    for k in range(1,6,2):
        newlandmarks[k] = landmarks[k]/height * 39
    #one dimension which represents one grey picture, set the first dimension as index
    x_data[i,:,:] = imagematrix
    y_data[i,:] = newlandmarks

for i in range(5849):                                               #train data part 2
    #get 39*39 numpy matrix of a single image
    imagename = train_table.cell(i+4152,0).value
    true_imagename = imagename[9:]
    imagematrix = nl.ImageToMatrix(IMAGESAVEURL_EN1_net + '\\' + true_imagename)
    #extract image size and rawlandmarks data for normalized newlandmarks
    img = Image.open(IMAGEURL + '\\' + imagename,'r')
    [width, height] = img.size
    rawlandmarks = train_table.row_slice(i+4152, start_colx=5, end_colx=11)
    #get ten normalized newlandmarks(coordinates of LE,RE,N,LM,RM)
    for l in range(6):
        landmarks[l] = rawlandmarks[l].value
    for j in range(0,5,2):
        newlandmarks[j] = landmarks[j]/width * 39
    for k in range(1,6,2):
        newlandmarks[k] = landmarks[k]/height * 39
    #one dimension which represents one grey picture, set the first dimension as index
    x_data[i+4151,:,:] = imagematrix
    y_data[i+4151,:] = newlandmarks  

## handle test data
for i in range(1439):                                               #test data part 1
    #get 39*39 numpy matrix of a single image
    imagename = test_table.cell(i+1,0).value
    true_imagename = imagename[9:]
    imagematrix = nl.ImageToMatrix(IMAGESAVEURL_EN1_lfw_test + '\\' + true_imagename)
    #extract image size and rawlandmarks data for normalized newlandmarks
    img = Image.open(IMAGEURL + '\\' + imagename,'r')
    [width, height] = img.size
    rawlandmarks = test_table.row_slice(i+1, start_colx=5, end_colx=11)
    #get ten normalized newlandmarks(coordinates of LE,RE,N,LM,RM)
    for l in range(6):
        landmarks[l] = rawlandmarks[l].value
    for j in range(0,5,2):
        newlandmarks[j] = landmarks[j]/width * 39
    for k in range(1,6,2):
        newlandmarks[k] = landmarks[k]/height * 39
    #one dimension which represents one grey picture, set the first dimension as index
    x_test[i,:,:] = imagematrix
    y_test[i,:] = newlandmarks

for i in range(2027):                                               #test data part 2
    #get 39*39 numpy matrix of a single image
    imagename = test_table.cell(i+1440,0).value
    true_imagename = imagename[9:]
    imagematrix = nl.ImageToMatrix(IMAGESAVEURL_EN1_net_test + '\\' + true_imagename)
    #extract image size and rawlandmarks data for normalized newlandmarks
    img = Image.open(IMAGEURL + '\\' + imagename,'r')
    [width, height] = img.size
    rawlandmarks = test_table.row_slice(i+1440, start_colx=5, end_colx=11)
    #get ten normalized newlandmarks(coordinates of LE,RE,N,LM,RM)
    for l in range(6):
        landmarks[l] = rawlandmarks[l].value
    for j in range(0,5,2):
        newlandmarks[j] = landmarks[j]/width * 39
    for k in range(1,6,2):
        newlandmarks[k] = landmarks[k]/height * 39
    #one dimension which represents one grey picture, set the first dimension as index
    x_test[i+1439,:,:] = imagematrix
    y_test[i+1439,:] = newlandmarks

## F1
x = tf.placeholder(tf.float32, shape=[None,31,39], name='x')        #input imagematrix_data to be fed
y = tf.placeholder(tf.float32, shape=[None,6], name='y')            #correct output to be fed
keep_prob = tf.placeholder(tf.float32, name='keep_prob')            #keep_prob parameter to be fed

x_image = tf.reshape(x, [-1,31,39,1])

## convolutional layer 1, kernel 4*4, insize 1, outsize 20
EN1_W_conv1 = nl.weight_variable([4,4,1,20])
EN1_b_conv1 = nl.bias_variable([20])
EN1_h_conv1 = nl.conv_layer(x_image, EN1_W_conv1) + EN1_b_conv1     #outsize = batch*28*36*20
EN1_a_conv1 = tf.nn.tanh(EN1_h_conv1)                               #outsize = batch*28*36*20

## max pooling layer 1
EN1_h_pool1 = nl.max_pool_22_layer(EN1_a_conv1)                     #outsize = batch*14*18*20
EN1_a_pool1 = tf.nn.tanh(EN1_h_pool1)                               #outsize = batch*14*18*20

## convolutional layer 2, kernel 3*3, insize 20, outsize 40
EN1_W_conv2 = nl.weight_variable([3,3,20,40])
EN1_b_conv2 = nl.bias_variable([40])
EN1_h_conv2 = nl.conv_layer(EN1_a_pool1, EN1_W_conv2) + EN1_b_conv2 #outsize = batch*12*16*40
EN1_a_conv2 = tf.nn.tanh(EN1_h_conv2)                               #outsize = batch*12*16*40

## max pooling layer 2
EN1_h_pool2 = nl.max_pool_22_layer(EN1_a_conv2)                     #outsize = batch*6*8*40
EN1_a_pool2 = tf.nn.tanh(EN1_h_pool2)                               #outsize = batch*6*8*40

## convolutional layer 3, kernel 3*3, insize 40, outsize 60
EN1_W_conv3 = nl.weight_variable([3,3,40,60])
EN1_b_conv3 = nl.bias_variable([60])
EN1_h_conv3 = nl.conv_layer(EN1_a_pool2, EN1_W_conv3) + EN1_b_conv3 #outsize = batch*4*6*60
EN1_a_conv3 = tf.nn.tanh(EN1_h_conv3)                               #outsize = batch*4*6*60

## max pooling layer 3
EN1_h_pool3 = nl.max_pool_22_layer(EN1_a_conv3)                     #outsize = batch*2*3*60
EN1_a_pool3 = tf.nn.tanh(EN1_h_pool3)                               #outsize = batch*2*3*60

## convolutional layer 4, kernel 2*2, insize 60, outsize 80
EN1_W_conv4 = nl.weight_variable([2,2,60,80])
EN1_b_conv4 = nl.bias_variable([80])
EN1_h_conv4 = nl.conv_layer(EN1_a_pool3, EN1_W_conv4) + EN1_b_conv4 #outsize = batch*1*2*80
EN1_a_conv4 = tf.nn.tanh(EN1_h_conv4)                               #outsize = batch*1*2*80

## flatten layer
EN1_x_flat = tf.reshape(EN1_a_conv4, [-1,160])                      #outsize = batch*160

## fully connected layer 1
EN1_W_fc1 = nl.weight_variable([160,100])
EN1_b_fc1 = nl.bias_variable([100])
EN1_h_fc1 = nl.fc_layer(EN1_x_flat, EN1_W_fc1, EN1_b_fc1)           #outsize = batch*100
EN1_a_fc1 = tf.nn.relu(EN1_h_fc1)                                   #outsize = batch*100
EN1_a_fc1_dropout = tf.nn.dropout(EN1_a_fc1, keep_prob)             #dropout layer 1

## fully connected layer 2
EN1_W_fc2 = nl.weight_variable([100,6])
EN1_b_fc2 = nl.bias_variable([6])
EN1_h_fc2 = nl.fc_layer(EN1_a_fc1_dropout, EN1_W_fc2, EN1_b_fc2)    #outsize = batch*6
EN1_a_fc2 = tf.nn.relu(EN1_h_fc2)                                   #outsize = batch*6

#regularization and loss function
original_cost = tf.reduce_mean(tf.pow(y - EN1_a_fc2, 2))
tv = tf.trainable_variables()   #L2 regularization
regularization_cost = 2 * tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])   #0.001 is hyperparameter
cost = original_cost + regularization_cost
Optimizer = tf.train.AdamOptimizer(0.005).minimize(cost)
init = tf.global_variables_initializer()
#average accuracy every batch
accuracy = tf.reduce_mean((y - EN1_a_fc2), 0)                       #average accuracy every batch
testaccuracy = np.zeros([27,6], dtype = np.float32)

with tf.Session() as sess:
    sess.run(init)
    for m in range(625):
        train_xbatch = x_data[(m*16):(m*16+16),:,:]                 #train 100 data every time, not including m*100+100
        train_ybatch = y_data[(m*16):(m*16+16),:]                   #train 100 data every time, not including m*100+100
        sess.run(Optimizer, feed_dict = {x:train_xbatch, y:train_ybatch, keep_prob:0.85})
        
    for n in range(27):
        test_xbatch = x_test[(n*128):(n*128+128),:,:]               #train 100 data every time, not including m*100+100
        test_ybatch = y_test[(n*128):(n*128+128),:]
        testaccuracy[n,:] = accuracy.eval(feed_dict = {x:test_xbatch, y:test_ybatch, keep_prob:1})

#print euclidean distance of the keypoints
printaccuracy = np.mean(testaccuracy, 0)
LE_accuracy = np.sqrt(np.square(printaccuracy[0])+np.square(printaccuracy[1])) / 39
RE_accuracy = np.sqrt(np.square(printaccuracy[2])+np.square(printaccuracy[3])) / 39
N_accuracy = np.sqrt(np.square(printaccuracy[4])+np.square(printaccuracy[5])) / 39
print('LE_error_rate is:',LE_accuracy)
print('RE_error_rate is:',RE_accuracy)
print('N_error_rate is:',N_accuracy)
