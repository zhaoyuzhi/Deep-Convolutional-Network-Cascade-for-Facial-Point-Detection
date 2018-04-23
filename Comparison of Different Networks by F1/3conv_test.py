# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 11:15:48 2018

@author: zhaoyuzhi
"""

from PIL import Image
import layer_definition as nl   #build a new layer and get ImageToMatrix function
import tensorflow as tf
import xlrd
import numpy as np
import pandas as pd

IMAGESAVEURL_F1_lfw = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_F1"
IMAGESAVEURL_F1_net = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_F1"
IMAGESAVEURL_F1_lfw_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_F1_test"
IMAGESAVEURL_F1_net_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_F1_test"
IMAGECASIA_test = "C:/Users/zhaoyuzhi/Desktop/CASIA_test"
train = xlrd.open_workbook('trainImageList.xlsx')
train_table = train.sheet_by_index(0)
CASIA_test_excel = xlrd.open_workbook('list.xlsx')
CASIA_test_table = CASIA_test_excel.sheet_by_index(0)

x_data = np.zeros([10000,39,39], dtype = np.float32)                #input imagematrix_data
y_data = np.ones([10000,10], dtype = np.float32)                    #correct output landmarks_data
CASIA_test = np.zeros([4442,39,39], dtype = np.float32)             #dataset to be processed

newlandmarks = np.zeros(10, dtype = np.float32)

## handle train data
for i in range(4151):                                               #train data part 1
    #get 39*39 numpy matrix of a single image
    imagename = train_table.cell(i+1,0).value
    true_imagename = imagename[9:]
    imagematrix = nl.ImageToMatrix(IMAGESAVEURL_F1_lfw + '\\' + true_imagename)
    #extract image size and rawlandmarks data for normalized newlandmarks
    width = train_table.cell(i+1,4).value - train_table.cell(i+1,3).value
    height = train_table.cell(i+1,2).value - train_table.cell(i+1,1).value
    rawlandmarks = train_table.row_slice(i+1, start_colx=5, end_colx=15)
    #get ten normalized newlandmarks(coordinates of LE,RE,N,LM,RM)
    for j in range(0,9,2):
        newlandmarks[j] = (rawlandmarks[j].value - train_table.cell(i+1,3).value + 0.05 * width) / (1.1 * width) * 39
    for k in range(1,10,2):
        newlandmarks[k] = (rawlandmarks[k].value - train_table.cell(i+1,1).value + 0.05 * height) / (1.1 * height) * 39
    #one dimension which represents one grey picture, set the first dimension as index
    x_data[i,:,:] = imagematrix
    y_data[i,:] = newlandmarks

for i in range(5849):                                               #train data part 2
    #get 39*39 numpy matrix of a single image
    imagename = train_table.cell(i+4152,0).value
    true_imagename = imagename[9:]
    imagematrix = nl.ImageToMatrix(IMAGESAVEURL_F1_net + '\\' + true_imagename)
    #extract image size and rawlandmarks data for normalized newlandmarks
    width = train_table.cell(i+4152,4).value - train_table.cell(i+4152,3).value
    height = train_table.cell(i+4152,2).value - train_table.cell(i+4152,1).value
    rawlandmarks = train_table.row_slice(i+4152, start_colx=5, end_colx=15)
    #get ten normalized newlandmarks(coordinates of LE,RE,N,LM,RM)
    for j in range(0,9,2):
        newlandmarks[j] = (rawlandmarks[j].value - train_table.cell(i+4152,3).value + 0.05 * width) / (1.1 * width) * 39
    for k in range(1,10,2):
        newlandmarks[k] = (rawlandmarks[k].value - train_table.cell(i+4152,1).value + 0.05 * height) / (1.1 * height) * 39
    #one dimension which represents one grey picture, set the first dimension as index
    x_data[i+4151,:,:] = imagematrix
    y_data[i+4151,:] = newlandmarks
    
# read actual test data
for i in range(4442):
    #get 39*39 numpy matrix of a single image
    imagename = CASIA_test_table.cell(i,0).value
    img = Image.open(IMAGECASIA_test + '/' + imagename,'r')
    img = img.crop([22,22,122,122])
    img = img.resize((39,39), Image.ANTIALIAS)
    imgdata = img.getdata()
    npdata = np.matrix(imgdata, dtype='float32') / 255.0
    newdata = np.reshape(npdata, (39,39))
    CASIA_test[i,:,:] = newdata

## define function for CNN
def weight_variable(shape):
    # Truncated normal distribution function
    # shape is kernel size, insize and outsize
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride = [1,x_movement,y_movement,1]
    # must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding='VALID')

def max_pool_22(x):
    # stride = [1,x_movement,y_movement,1]
    # must have strides[0] = strides[3] = 1
    # ksize(kernel size) = [1,length,height,1]
    return tf.nn.max_pool(x, ksize = [1,2,2,1],
                          strides = [1,2,2,1], padding='SAME')

## define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None,39,39])    #39*39
ys = tf.placeholder(tf.float32, [None,10])       #the true value of y
keep_prob = tf.placeholder(tf.float32)
# -1 as default number of images, 1 as gray image
x_image = tf.reshape(xs, [-1,39,39,1])
# print(x_image.shape) [n_sanmples,39,39,1]

## convolutional layer 1, kernel 5*5, insize 1, outsize 32
# the first/second parameter are the sizes of kernel, the third parameter
# is the channel of images, the fourth parameter is feature map
W_conv1 = weight_variable([4,4,1,20])
b_conv1 = bias_variable([20])
h_conv1 = conv2d(x_image, W_conv1) + b_conv1                        #outsize = batch*36*36*20
a_conv1 = tf.nn.relu(h_conv1)                                       #outsize = batch*36*36*20
# max pooling layer 1
h_pool1 = max_pool_22(a_conv1)                                      #outsize = batch*18*18*20
a_pool1 = tf.nn.relu(h_pool1)                                       #outsize = batch*18*18*20
# convolutional layer 2
W_conv2 = weight_variable([5,5,20,40])
b_conv2 = bias_variable([40])
h_conv2 = conv2d(a_pool1, W_conv2) + b_conv2                        #outsize = batch*14*14*40
a_conv2 = tf.nn.relu(h_conv2)                                       #outsize = batch*14*14*40
# max pooling layer 2
h_pool2 = max_pool_22(a_conv2)                                      #outsize = batch*7*7*40
a_pool2 = tf.nn.relu(h_pool2)                                       #outsize = batch*7*7*40
# convolutional layer 3
W_conv3 = weight_variable([4,4,40,80])
b_conv3 = bias_variable([80])
h_conv3 = conv2d(a_pool2, W_conv3) + b_conv3                        #outsize = batch*4*4*80
a_conv3 = tf.nn.relu(h_conv3)                                       #outsize = batch*4*4*80
# flat
a_conv3_flat = tf.reshape(a_conv3, [-1,4*4*80])                     #[batch,4,4,80] = [batch,4*4*80]
# fully connected layer 1
W_fc1 = weight_variable([4*4*80,1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.matmul(a_conv3_flat, W_fc1) + b_fc1
a_fc1 = tf.nn.relu(h_fc1)
# dropout for fc layer1
a_fc1_drop = tf.nn.dropout(a_fc1, keep_prob)
# fully connected layer 2
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
h_fc2 = tf.matmul(a_fc1_drop, W_fc2) + b_fc2
prediction = tf.nn.relu(h_fc2)                                      #the prediction value of y

## define loss and accuracy
loss = tf.reduce_mean(tf.pow(ys - prediction, 2))
Optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)
cache_F1 = np.zeros([4442,10], dtype = np.float32)

## start training
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # train 300 times
    for i in range(500):                                             #iteration times
        for m in range(625):                                        #training process using training data 10000 images
            train_xbatch = x_data[(m*16):(m*16+16),:,:]             #train 16 data every batch, not including m*16+16
            train_ybatch = y_data[(m*16):(m*16+16),:]               #train 16 data every batch, not including m*16+16
            sess.run(Optimizer, feed_dict = {xs:train_xbatch, ys:train_ybatch, keep_prob:0.5})
    '''
    # test trained model
    for k in range(27):
        test_xbatch = x_test[(k*128):(k*128+128),:,:]               #train 128 data every time, not including m*100+100
        test_ybatch = y_test[(k*128):(k*128+128),:]
        cache_F1[(k*128):(k*128+128),:] = prediction.eval(feed_dict = {xs:test_xbatch, ys:test_ybatch, keep_prob:1})
    '''
    # test trained model
    for k in range(2221):
        test_xbatch = CASIA_test[(k*2):(k*2+2),:,:]                 #test 2 data every batch
        cache_F1[(k*2):(k*2+2),:] = prediction.eval(feed_dict = {xs:test_xbatch, keep_prob:1})

print(cache_F1)
## save the predicted keypoints to excel
cache_F1_df = pd.DataFrame(cache_F1)
writer = pd.ExcelWriter('test_3conv.xlsx')
cache_F1_df.to_excel(writer,'sheet1')
writer.save()