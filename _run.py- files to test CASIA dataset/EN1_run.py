# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 17:05:01 2018

@author: zhaoyuzhi
"""

from PIL import Image
import layer_definition as nl   #build a new layer and get ImageToMatrix function
import tensorflow as tf
import xlrd
import numpy as np
import pandas as pd

IMAGESAVEURL_EN1_lfw = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_EN1"
IMAGESAVEURL_EN1_net = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_EN1"
IMAGESAVEURL_EN1_lfw_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_EN1_test"
IMAGESAVEURL_EN1_net_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_EN1_test"
IMAGECASIA_test = "C:/Users/zhaoyuzhi/Desktop/CASIA_test"
train = xlrd.open_workbook('trainImageList.xlsx')
train_table = train.sheet_by_index(0)
CASIA_test_excel = xlrd.open_workbook('list.xlsx')
CASIA_test_table = CASIA_test_excel.sheet_by_index(0)
x_data = np.zeros([10000,31,39], dtype = np.float32)                #input imagematrix_data
y_data = np.ones([10000,6], dtype = np.float32)                     #correct output landmarks_data
CASIA_test = np.zeros([4442,31,39], dtype = np.float32)             #dataset to be processed

newlandmarks = np.zeros(6, dtype = np.float32)

## handle train data
for i in range(4151):                                               #train data part 1
    #get 39*39 numpy matrix of a single image
    imagename = train_table.cell(i+1,0).value
    true_imagename = imagename[9:]
    imagematrix = nl.ImageToMatrix(IMAGESAVEURL_EN1_lfw + '\\' + true_imagename)
    #extract image size and rawlandmarks data for normalized newlandmarks
    width = train_table.cell(i+1,4).value - train_table.cell(i+1,3).value
    height = train_table.cell(i+1,2).value - train_table.cell(i+1,1).value
    rawlandmarks = train_table.row_slice(i+1, start_colx=5, end_colx=11)
    #get ten normalized newlandmarks(coordinates of LE,RE,N,LM,RM)
    for j in range(0,5,2):
        newlandmarks[j] = (rawlandmarks[j].value - train_table.cell(i+1,3).value + 0.05 * width) / (1.1 * width) * 39
    for k in range(1,6,2):
        newlandmarks[k] = (rawlandmarks[k].value - train_table.cell(i+1,1).value + 0.04 * height) / (0.88 * height) * 39
    #one dimension which represents one grey picture, set the first dimension as index
    x_data[i,:,:] = imagematrix
    y_data[i,:] = newlandmarks

for i in range(5849):                                               #train data part 2
    #get 39*39 numpy matrix of a single image
    imagename = train_table.cell(i+4152,0).value
    true_imagename = imagename[9:]
    imagematrix = nl.ImageToMatrix(IMAGESAVEURL_EN1_net + '\\' + true_imagename)
    #extract image size and rawlandmarks data for normalized newlandmarks
    width = train_table.cell(i+4152,4).value - train_table.cell(i+4152,3).value
    height = train_table.cell(i+4152,2).value - train_table.cell(i+4152,1).value
    rawlandmarks = train_table.row_slice(i+4152, start_colx=5, end_colx=11)
    #get ten normalized newlandmarks(coordinates of LE,RE,N,LM,RM)
    for j in range(0,5,2):
        newlandmarks[j] = (rawlandmarks[j].value - train_table.cell(i+4152,3).value + 0.05 * width) / (1.1 * width) * 39
    for k in range(1,6,2):
        newlandmarks[k] = (rawlandmarks[k].value - train_table.cell(i+4152,1).value + 0.04 * height) / (0.88 * height) * 39
    #one dimension which represents one grey picture, set the first dimension as index
    x_data[i+4151,:,:] = imagematrix
    y_data[i+4151,:] = newlandmarks

# read actual test data
for i in range(4442):
    #get 31*39 numpy matrix of a single image
    #input images (width = 144, height = 144)
    #according to 'F1_500.xlsx', min(LEx) = 35.45, min(LEy) = 37.60, max(REx) = 112.21, min(REy) = 37.93
    #according to 'F1_500.xlsx', the range of Nx is [39.12,96.26], the range of Ny is [60.43,95.46]
    #conclusion: min(left) = 35, min(top) = 37, max(right) = 113, max(bottom) = 96
    #at first crop a square region for test, width is 80 and height is 64 (approximately 31:39)
    #that is faceboundingbox = [35,59,115,123]
    #output/crop face_bounding_box(left,top,right,bottom)
    #img.resize((width, height))
    imagename = CASIA_test_table.cell(i,0).value
    img = Image.open(IMAGECASIA_test + '/' + imagename,'r')
    img = img.crop([22,22,122,110])
    img_resize = img.resize((39,31), Image.ANTIALIAS)
    imgdata = img_resize.getdata()
    npdata = np.matrix(imgdata, dtype= np.float32) / 255.0
    newdata = np.reshape(npdata, (31,39))
    CASIA_test[i,:,:] = newdata

## F1
x = tf.placeholder(tf.float32, shape=[None,31,39], name='x')        #input imagematrix_data to be fed
y = tf.placeholder(tf.float32, shape=[None,6], name='y')            #correct output to be fed
keep_prob = tf.placeholder(tf.float32, name='keep_prob')            #keep_prob parameter to be fed

x_image = tf.reshape(x, [-1,31,39,1])

## convolutional layer 1, kernel 4*4, insize 1, outsize 20
EN1_W_conv1 = nl.weight_variable([4,4,1,20])
EN1_b_conv1 = nl.bias_variable([20])
EN1_h_conv1 = nl.conv_layer(x_image, EN1_W_conv1) + EN1_b_conv1     #outsize = batch*28*36*20
EN1_a_conv1 = tf.nn.relu(EN1_h_conv1)                               #outsize = batch*28*36*20

## max pooling layer 1
EN1_h_pool1 = nl.max_pool_22_layer(EN1_a_conv1)                     #outsize = batch*14*18*20
EN1_a_pool1 = tf.nn.relu(EN1_h_pool1)                               #outsize = batch*14*18*20

## convolutional layer 2, kernel 3*3, insize 20, outsize 40
EN1_W_conv2 = nl.weight_variable([3,3,20,40])
EN1_b_conv2 = nl.bias_variable([40])
EN1_h_conv2 = nl.conv_layer(EN1_a_pool1, EN1_W_conv2) + EN1_b_conv2 #outsize = batch*12*16*40
EN1_a_conv2 = tf.nn.relu(EN1_h_conv2)                               #outsize = batch*12*16*40

## max pooling layer 2
EN1_h_pool2 = nl.max_pool_22_layer(EN1_a_conv2)                     #outsize = batch*6*8*40
EN1_a_pool2 = tf.nn.relu(EN1_h_pool2)                               #outsize = batch*6*8*40

## convolutional layer 3, kernel 3*3, insize 40, outsize 60
EN1_W_conv3 = nl.weight_variable([3,3,40,60])
EN1_b_conv3 = nl.bias_variable([60])
EN1_h_conv3 = nl.conv_layer(EN1_a_pool2, EN1_W_conv3) + EN1_b_conv3 #outsize = batch*4*6*60
EN1_a_conv3 = tf.nn.relu(EN1_h_conv3)                               #outsize = batch*4*6*60

## max pooling layer 3
EN1_h_pool3 = nl.max_pool_22_layer(EN1_a_conv3)                     #outsize = batch*2*3*60
EN1_a_pool3 = tf.nn.relu(EN1_h_pool3)                               #outsize = batch*2*3*60

## convolutional layer 4, kernel 2*2, insize 60, outsize 80
EN1_W_conv4 = nl.weight_variable([2,2,60,80])
EN1_b_conv4 = nl.bias_variable([80])
EN1_h_conv4 = nl.conv_layer(EN1_a_pool3, EN1_W_conv4) + EN1_b_conv4 #outsize = batch*1*2*80
EN1_a_conv4 = tf.nn.relu(EN1_h_conv4)                               #outsize = batch*1*2*80

## flatten layer
EN1_x_flat = tf.reshape(EN1_a_conv4, [-1,160])                      #outsize = batch*160

## fully connected layer 1
EN1_W_fc1 = nl.weight_variable([160,100])
EN1_b_fc1 = nl.bias_variable([100])
EN1_h_fc1 = tf.matmul(EN1_x_flat, EN1_W_fc1) + EN1_b_fc1            #outsize = batch*100
EN1_a_fc1 = tf.nn.relu(EN1_h_fc1)                                   #outsize = batch*100
EN1_a_fc1_dropout = tf.nn.dropout(EN1_a_fc1, keep_prob)             #dropout layer 1

## fully connected layer 2
EN1_W_fc2 = nl.weight_variable([100,6])
EN1_b_fc2 = nl.bias_variable([6])
EN1_h_fc2 = tf.matmul(EN1_a_fc1_dropout, EN1_W_fc2) + EN1_b_fc2     #outsize = batch*6
EN1_a_fc2 = tf.nn.relu(EN1_h_fc2)                                   #outsize = batch*6

#regularization and loss function
original_cost = tf.reduce_mean(tf.pow(y - EN1_a_fc2, 2))
tv = tf.trainable_variables()   #L2 regularization
regularization_cost = 2 * tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])   #0.001 is hyperparameter
cost = original_cost + regularization_cost
Optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)
init = tf.global_variables_initializer()
cache_EN1 = np.zeros([4442,6], dtype = np.float32)

with tf.Session() as sess:
    sess.run(init)
    for i in range(500):                                            #number of iterations:500*625=312500, 5 millon images
        for m in range(625):                                        #training process using training data 10000 images
            train_xbatch = x_data[(m*16):(m*16+16),:,:]             #train 16 data every batch, not including m*16+16
            train_ybatch = y_data[(m*16):(m*16+16),:]               #train 16 data every batch, not including m*16+16
            sess.run(Optimizer, feed_dict = {x:train_xbatch, y:train_ybatch, keep_prob:0.5})
    
    for k in range(2221):
        test_xbatch = CASIA_test[(k*2):(k*2+2),:,:]                 #train 2 data every batch, not including m*2+2
        cache_EN1[(k*2):(k*2+2),:] = EN1_a_fc2.eval(feed_dict = {x:test_xbatch, keep_prob:1})

print(cache_EN1)

## save the predicted keypoints to excel
cache_EN1_df = pd.DataFrame(cache_EN1)
writer = pd.ExcelWriter('EN1.xlsx')
cache_EN1_df.to_excel(writer,'sheet1')
writer.save()
