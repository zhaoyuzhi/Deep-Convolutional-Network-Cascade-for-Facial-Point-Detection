# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 09:40:05 2018

@author: zhaoyuzhi
"""

from PIL import Image
import layer_definition as nl   #build a new layer and get ImageToMatrix function
import tensorflow as tf
import math
import random
import xlrd
import numpy as np

IMAGEURL = "C:\\Users\\zhaoyuzhi\\Desktop\\train"
train = xlrd.open_workbook('trainImageList.xlsx')
train_table = train.sheet_by_index(0)
test = xlrd.open_workbook('testImageList.xlsx')
test_table = test.sheet_by_index(0)
x_data = np.zeros([100000,15,15], dtype = np.float32)               #input imagematrix_data 10*10000 (10 is range of 'j')
y_data = np.ones([100000,2], dtype = np.float32)                    #correct output landmarks_data
x_test = np.zeros([34660,15,15], dtype = np.float32)                #10*3466
y_test = np.ones([34660,2], dtype = np.float32)                     #10*3466

newlandmarks = np.zeros(2, dtype = np.float32)

## handle train data
for i in range(4151):                                               #train data part 1
    #get 15*15 numpy matrix of a single image
    imagename = train_table.cell(i+1,0).value
    img = Image.open(IMAGEURL + '\\' + imagename,'r')               #the whole image
    img = img.convert('L')
    Px = train_table.cell(i+1,11).value                             #x of the ground truth position
    Py = train_table.cell(i+1,12).value                             #y of the ground truth position
    delta_x = math.ceil(0.18*(train_table.cell(i+1,4).value - train_table.cell(i+1,3).value))
    delta_y = math.ceil(0.18*(train_table.cell(i+1,2).value - train_table.cell(i+1,1).value))
    for j in range(10):                                             #as for one same image, increasing the random situation
        rx = -0.2777 + 0.555 * random.random()                      #output random number [-1,1], 0.05 / 0.18 = 0.2777
        ry = -0.2777 + 0.555 * random.random()                      #output random number [-1,1], 0.05 / 0.18 = 0.2777
        newleft = Px + (rx-1) * delta_x                             #level 2 bounding box parameter
        newright = Px + (rx+1) * delta_x                            #level 2 bounding box parameter
        newtop = Py + (ry-1) * delta_y                              #level 2 bounding box parameter
        newbottom = Py + (ry+1) * delta_y                           #level 2 bounding box parameter
        faceboundingbox = [newleft, newtop, newright, newbottom]
        region = img.crop(faceboundingbox)
        imagematrix = nl.Img15ToMatrix(region)
        #get two normalized newlandmarks, which is the ratio of level 2 bounding box
        newlandmarks[0] = (1-rx) / 2 * 39                           #ratio x
        newlandmarks[1] = (1-ry) / 2 * 39                           #ratio y
        #one dimension which represents one grey picture, set the first dimension as index
        x_data[i*10+j,:,:] = imagematrix
        y_data[i*10+j,:] = newlandmarks

for i in range(5849):                                               #train data part 2
    #get 15*15 numpy matrix of a single image
    imagename = train_table.cell(i+4152,0).value
    img = Image.open(IMAGEURL + '\\' + imagename,'r')               #the whole image
    img = img.convert('L')
    Px = train_table.cell(i+4152,11).value                          #x of the ground truth position
    Py = train_table.cell(i+4152,12).value                          #y of the ground truth position
    delta_x = math.ceil(0.18*(train_table.cell(i+4152,4).value - train_table.cell(i+4152,3).value))
    delta_y = math.ceil(0.18*(train_table.cell(i+4152,2).value - train_table.cell(i+4152,1).value))
    for j in range(10):                                             #as for one same image, increasing the random situation
        rx = -0.2777 + 0.555 * random.random()                      #output random number [-1,1], 0.05 / 0.18 = 0.2777
        ry = -0.2777 + 0.555 * random.random()                      #output random number [-1,1], 0.05 / 0.18 = 0.2777
        newleft = Px + (rx-1) * delta_x                             #level 2 bounding box parameter
        newright = Px + (rx+1) * delta_x                            #level 2 bounding box parameter
        newtop = Py + (ry-1) * delta_y                              #level 2 bounding box parameter
        newbottom = Py + (ry+1) * delta_y                           #level 2 bounding box parameter
        faceboundingbox = [newleft, newtop, newright, newbottom]
        region = img.crop(faceboundingbox)
        imagematrix = nl.Img15ToMatrix(region)
        #get two normalized newlandmarks, which is the ratio of level 2 bounding box
        newlandmarks[0] = (1-rx) / 2 * 39                           #ratio x
        newlandmarks[1] = (1-ry) / 2 * 39                           #ratio y
        #one dimension which represents one grey picture, set the first dimension as index
        x_data[41510+i*10+j,:,:] = imagematrix
        y_data[41510+i*10+j,:] = newlandmarks

## handle test data
for i in range(1439):                                               #test data part 1
    #get 15*15 numpy matrix of a single image
    imagename = test_table.cell(i+1,0).value
    img = Image.open(IMAGEURL + '\\' + imagename,'r')               #the whole image
    img = img.convert('L')
    Px = test_table.cell(i+1,11).value                              #x of the ground truth position
    Py = test_table.cell(i+1,12).value                              #y of the ground truth position
    delta_x = math.ceil(0.18*(test_table.cell(i+1,4).value - test_table.cell(i+1,3).value))
    delta_y = math.ceil(0.18*(test_table.cell(i+1,2).value - test_table.cell(i+1,1).value))
    for j in range(10):                                             #as for one same image, increasing the random situation
        rx = -0.2777 + 0.555 * random.random()                      #output random number [-1,1], 0.05 / 0.18 = 0.2777
        ry = -0.2777 + 0.555 * random.random()                      #output random number [-1,1], 0.05 / 0.18 = 0.2777
        newleft = Px + (rx-1) * delta_x                             #level 2 bounding box parameter
        newright = Px + (rx+1) * delta_x                            #level 2 bounding box parameter
        newtop = Py + (ry-1) * delta_y                              #level 2 bounding box parameter
        newbottom = Py + (ry+1) * delta_y                           #level 2 bounding box parameter
        faceboundingbox = [newleft, newtop, newright, newbottom]
        region = img.crop(faceboundingbox)
        imagematrix = nl.Img15ToMatrix(region)
        #get two normalized newlandmarks, which is the ratio of level 2 bounding box
        newlandmarks[0] = (1-rx) / 2 * 39                           #ratio x
        newlandmarks[1] = (1-ry) / 2 * 39                           #ratio y
        #one dimension which represents one grey picture, set the first dimension as index
        x_test[i*10+j,:,:] = imagematrix
        y_test[i*10+j,:] = newlandmarks

for i in range(2027):                                               #test data part 2
    #get 15*15 numpy matrix of a single image
    imagename = test_table.cell(i+1440,0).value
    img = Image.open(IMAGEURL + '\\' + imagename,'r')               #the whole image
    img = img.convert('L')
    Px = test_table.cell(i+1440,11).value                           #x of the ground truth position
    Py = test_table.cell(i+1440,12).value                           #y of the ground truth position
    delta_x = math.ceil(0.18*(test_table.cell(i+1440,4).value - test_table.cell(i+1440,3).value))
    delta_y = math.ceil(0.18*(test_table.cell(i+1440,2).value - test_table.cell(i+1440,1).value))
    for j in range(10):                                             #as for one same image, increasing the random situation
        rx = -0.2777 + 0.555 * random.random()                      #output random number [-1,1], 0.05 / 0.18 = 0.2777
        ry = -0.2777 + 0.555 * random.random()                      #output random number [-1,1], 0.05 / 0.18 = 0.2777
        newleft = Px + (rx-1) * delta_x                             #level 2 bounding box parameter
        newright = Px + (rx+1) * delta_x                            #level 2 bounding box parameter
        newtop = Py + (ry-1) * delta_y                              #level 2 bounding box parameter
        newbottom = Py + (ry+1) * delta_y                           #level 2 bounding box parameter
        faceboundingbox = [newleft, newtop, newright, newbottom]
        region = img.crop(faceboundingbox)
        imagematrix = nl.Img15ToMatrix(region)
        #get two normalized newlandmarks, which is the ratio of level 2 bounding box
        newlandmarks[0] = (1-rx) / 2 * 39                           #ratio x
        newlandmarks[1] = (1-ry) / 2 * 39                           #ratio y
        #one dimension which represents one grey picture, set the first dimension as index
        x_test[14390+i*10+j,:,:] = imagematrix
        y_test[14390+i*10+j,:] = newlandmarks

## LM22
x = tf.placeholder(tf.float32, shape=[None,15,15], name='x')        #input imagematrix_data to be fed
y = tf.placeholder(tf.float32, shape=[None,2], name='y')            #correct output to be fed
keep_prob = tf.placeholder(tf.float32, name='keep_prob')            #keep_prob parameter to be fed

x_image = tf.reshape(x, [-1,15,15,1])

## convolutional layer 1, kernel 4*4, insize 1, outsize 20
W_conv1 = nl.weight_variable([4,4,1,20])
b_conv1 = nl.bias_variable([20])
h_conv1 = nl.conv_layer(x_image, W_conv1) + b_conv1                 #outsize = batch*12*12*20
a_conv1 = tf.nn.relu(h_conv1)                                       #outsize = batch*12*12*20

## max pooling layer 1
h_pool1 = nl.max_pool_22_layer(a_conv1)                             #outsize = batch*6*6*20
a_pool1 = tf.nn.relu(h_pool1)                                       #outsize = batch*6*6*20

## convolutional layer 2, kernel 3*3, insize 20, outsize 40
W_conv2 = nl.weight_variable([3,3,20,40])
b_conv2 = nl.bias_variable([40])
h_conv2 = nl.conv_layer(a_pool1, W_conv2) + b_conv2                 #outsize = batch*4*4*40
a_conv2 = tf.nn.relu(h_conv2)                                       #outsize = batch*4*4*40

## max pooling layer 2
h_pool2 = nl.max_pool_22_layer(a_conv2)                             #outsize = batch*2*2*40
a_pool2 = tf.nn.relu(h_pool2)                                       #outsize = batch*2*2*40

## flatten layer
x_flat = tf.reshape(a_pool2, [-1,160])                              #outsize = batch*160

## fully connected layer 1
W_fc1 = nl.weight_variable([160,60])
b_fc1 = nl.bias_variable([60])
h_fc1 = tf.matmul(x_flat, W_fc1) + b_fc1                            #outsize = batch*60
a_fc1 = tf.nn.relu(h_fc1)                                           #outsize = batch*60
a_fc1_dropout = tf.nn.dropout(a_fc1, keep_prob)                     #dropout layer 1

## fully connected layer 2
W_fc2 = nl.weight_variable([60,2])
b_fc2 = nl.bias_variable([2])
h_fc2 = tf.matmul(a_fc1_dropout, W_fc2) + b_fc2                     #outsize = batch*10
a_fc2 = tf.nn.relu(h_fc2)                                           #outsize = batch*10

#regularization and loss function
original_cost = tf.reduce_mean(tf.pow(y - a_fc2, 2))
tv = tf.trainable_variables()   #L2 regularization
regularization_cost = 2 * tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])   #2 is hyperparameter
cost = original_cost + regularization_cost
Optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)
init = tf.global_variables_initializer()
cache_LM22= np.zeros([4442,2], dtype = np.float32)                 #run CASIA_test dataset

with tf.Session() as sess:
    sess.run(init)
    for i in range(50):                                             #iteration times:50*6250=312500, 5 millon images
        for m in range(6250):                                       #training process using training data 10000 images
            train_xbatch = x_data[(m*16):(m*16+16),:,:]             #train 16 data every batch, not including m*16+16
            train_ybatch = y_data[(m*16):(m*16+16),:]               #train 16 data every batch, not including m*16+16
            sess.run(Optimizer, feed_dict = {x:train_xbatch, y:train_ybatch, keep_prob:0.5})

    for k in range(17330):
        test_xbatch = x_test[(k*2):(k*2+2),:,:]                     #test 2 data every time, not including k*2+2
        test_ybatch = y_test[(k*2):(k*2+2),:]
        testaccuracy[k,:] = accuracy.eval(feed_dict = {x:test_xbatch, y:test_ybatch, keep_prob:1})
        cache_LM22[(k*2):(k*2+2),:] = a_fc2.eval(feed_dict = {x:test_xbatch, y:test_ybatch, keep_prob:1})

#print euclidean distance of the keypoints
printaccuracy = np.mean(testaccuracy, 0)
LM_accuracy = np.sqrt(np.square(printaccuracy[0])+np.square(printaccuracy[1])) / 39
print('LM_error_rate is:',LM_accuracy)
print('output_test_landmarks is:',cache_LM22)
