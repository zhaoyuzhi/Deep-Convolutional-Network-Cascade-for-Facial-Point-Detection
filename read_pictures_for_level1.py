# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 12:18:05 2018

@author: zhaoyuzhi
"""

#INDEED WE CAN BUILD THIS PROGRAM AND NEURAL NETWORK PART TOGETHER, BUT WE WANT TO SEE THE PROCESSING IMAGE
#AND IT IS EASY FOR US TO TRAIN LEVEL 1 NETWORK
from PIL import Image
import math
import xlrd

#read LFPW images according to excel file
IMAGEURL = "C:\\Users\\zhaoyuzhi\\Desktop\\train"

IMAGESAVEURL_F1_lfw = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_F1"
IMAGESAVEURL_EN1_lfw = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_EN1"
IMAGESAVEURL_NM1_lfw = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_NM1"
IMAGESAVEURL_box_lfw = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_face_bounding_box"

IMAGESAVEURL_F1_net = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_F1"
IMAGESAVEURL_EN1_net = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_EN1"
IMAGESAVEURL_NM1_net = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_NM1"
IMAGESAVEURL_box_net = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_face_bounding_box"

IMAGESAVEURL_F1_lfw_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_F1_test"
IMAGESAVEURL_EN1_lfw_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_EN1_test"
IMAGESAVEURL_NM1_lfw_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_NM1_test"
IMAGESAVEURL_box_lfw_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_face_bounding_box_test"

IMAGESAVEURL_F1_net_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_F1_test"
IMAGESAVEURL_EN1_net_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_EN1_test"
IMAGESAVEURL_NM1_net_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_NM1_test"
IMAGESAVEURL_box_net_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_face_bounding_box_test"

lfwtestlib = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfwtestlib"     #for a test

train = xlrd.open_workbook('trainImageList.xlsx')
table = train.sheet_by_index(0)
test = xlrd.open_workbook('testImageList.xlsx')
test_table = test.sheet_by_index(0)

#this is a test produced by author 'zhaoyuzhi', you should not run this code
'''
for i in range(4151):
    imagename = table.cell(i+1,0).value
    true_imagename = imagename[9:]
    img = Image.open(IMAGEURL + '\\' + imagename,'r')
    delta_x = math.ceil(0.05*(table.cell(i+1,4).value - table.cell(i+1,3).value))
    delta_NM1top = math.ceil(0.18*(table.cell(i+1,2).value - table.cell(i+1,1).value))
    delta_NM1bottom = math.ceil(0.05*(table.cell(i+1,2).value - table.cell(i+1,1).value))
    faceboundingbox_NM1 = [(table.cell(i+1,3).value - delta_x), (table.cell(i+1,1).value + delta_NM1top),
                   (table.cell(i+1,4).value + delta_x), (table.cell(i+1,2).value + delta_NM1bottom)]
    region_NM1_resize = img.crop(faceboundingbox_NM1)
    #region_NM1_resize = region_NM1.resize((39,31), Image.ANTIALIAS)   #EN1 image
    region_NM1_resize_grey = region_NM1_resize.convert('L')   #convert to grey image
    region_NM1_resize_grey.save(lfwtestlib + '\\' + true_imagename)
'''

#this is for lfw_5590 train
'''
for i in range(4151):
    imagename = table.cell(i+1,0).value
    true_imagename = imagename[9:]
    img = Image.open(IMAGEURL + '\\' + imagename,'r')
    #excel face_bounding_box(top,bottom,left,right)
    #output/crop face_bounding_box(left,top,right,bottom)
    #img.resize((width, height))
    #for F1:
    delta_x = math.ceil(0.05*(table.cell(i+1,4).value - table.cell(i+1,3).value))
    delta_y = math.ceil(0.05*(table.cell(i+1,2).value - table.cell(i+1,1).value))
    faceboundingbox = [(table.cell(i+1,3).value - delta_x), (table.cell(i+1,1).value - delta_y),
                   (table.cell(i+1,4).value + delta_x), (table.cell(i+1,2).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    ## region_grey = region.convert('L')   #convert to grey image
    ## region_grey.save(IMAGESAVEURL_box_lfw + '\\' + true_imagename)
    region_resize = region.resize((39,39), Image.ANTIALIAS)   #F1 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_F1_lfw + '\\' + true_imagename)
    #for EN1:
    delta_EN1top = math.ceil(0.04*(table.cell(i+1,2).value - table.cell(i+1,1).value))
    delta_EN1bottom = math.ceil(0.84*(table.cell(i+1,2).value - table.cell(i+1,1).value))
    faceboundingbox_EN1 = [(table.cell(i+1,3).value - delta_x), (table.cell(i+1,1).value - delta_EN1top),
                   (table.cell(i+1,4).value + delta_x), (table.cell(i+1,1).value + delta_EN1bottom)]
    region_EN1 = img.crop(faceboundingbox_EN1)
    region_EN1_resize = region_EN1.resize((39,31), Image.ANTIALIAS)   #EN1 image
    region_EN1_resize_grey = region_EN1_resize.convert('L')   #convert to grey image
    region_EN1_resize_grey.save(IMAGESAVEURL_EN1_lfw + '\\' + true_imagename)
    #for NM1:
    delta_NM1top = math.ceil(0.18*(table.cell(i+1,2).value - table.cell(i+1,1).value))
    delta_NM1bottom = math.ceil(1.05*(table.cell(i+1,2).value - table.cell(i+1,1).value))
    faceboundingbox_NM1 = [(table.cell(i+1,3).value - delta_x), (table.cell(i+1,1).value + delta_NM1top),
                   (table.cell(i+1,4).value + delta_x), (table.cell(i+1,1).value + delta_NM1bottom)]
    region_NM1 = img.crop(faceboundingbox_NM1)
    region_NM1_resize = region_NM1.resize((39,31), Image.ANTIALIAS)   #EN1 image
    region_NM1_resize_grey = region_NM1_resize.convert('L')   #convert to grey image
    region_NM1_resize_grey.save(IMAGESAVEURL_NM1_lfw + '\\' + true_imagename)
'''

#this is for net_7876 train
'''
for i in range(5849):
    imagename = table.cell(i+4152,0).value
    true_imagename = imagename[9:]
    img = Image.open(IMAGEURL + '\\' + imagename,'r')
    #excel face_bounding_box(top,bottom,left,right)
    #output/crop face_bounding_box(left,top,right,bottom)
    #img.resize((width, height))
    #for F1:
    delta_x = math.ceil(0.05*(table.cell(i+4152,4).value - table.cell(i+4152,3).value))
    delta_y = math.ceil(0.05*(table.cell(i+4152,2).value - table.cell(i+4152,1).value))
    faceboundingbox = [(table.cell(i+4152,3).value - delta_x), (table.cell(i+4152,1).value - delta_y),
                   (table.cell(i+4152,4).value + delta_x), (table.cell(i+4152,2).value + delta_y)]
    region = img.crop(faceboundingbox)
    ## region_grey = region.convert('L')
    ## region_grey.save(IMAGESAVEURL_box_net + '\\' + true_imagename)
    region_resize = region.resize((39,39), Image.ANTIALIAS)
    region_resize_grey = region_resize.convert('L')
    region_resize_grey.save(IMAGESAVEURL_F1_net + '\\' + true_imagename)
    #for EN1:
    delta_EN1top = math.ceil(0.04*(table.cell(i+4152,2).value - table.cell(i+4152,1).value))
    delta_EN1bottom = math.ceil(0.84*(table.cell(i+4152,2).value - table.cell(i+4152,1).value))
    faceboundingbox_EN1 = [(table.cell(i+4152,3).value - delta_x), (table.cell(i+4152,1).value - delta_EN1top),
                   (table.cell(i+4152,4).value + delta_x), (table.cell(i+4152,1).value + delta_EN1bottom)]
    region_EN1 = img.crop(faceboundingbox_EN1)
    region_EN1_resize = region_EN1.resize((39,31), Image.ANTIALIAS)   #EN1 image
    region_EN1_resize_grey = region_EN1_resize.convert('L')   #convert to grey image
    region_EN1_resize_grey.save(IMAGESAVEURL_EN1_net + '\\' + true_imagename)
    #for NM1:
    delta_NM1top = math.ceil(0.18*(table.cell(i+4152,2).value - table.cell(i+4152,1).value))
    delta_NM1bottom = math.ceil(1.05*(table.cell(i+4152,2).value - table.cell(i+4152,1).value))
    faceboundingbox_NM1 = [(table.cell(i+4152,3).value - delta_x), (table.cell(i+4152,1).value + delta_NM1top),
                   (table.cell(i+4152,4).value + delta_x), (table.cell(i+4152,1).value + delta_NM1bottom)]
    region_NM1 = img.crop(faceboundingbox_NM1)
    region_NM1_resize = region_NM1.resize((39,31), Image.ANTIALIAS)   #EN1 image
    region_NM1_resize_grey = region_NM1_resize.convert('L')   #convert to grey image
    region_NM1_resize_grey.save(IMAGESAVEURL_NM1_net + '\\' + true_imagename)
'''

#this is for lfw_5590 test
'''
for i in range(1439):
    imagename = test_table.cell(i+1,0).value
    true_imagename = imagename[9:]
    img = Image.open(IMAGEURL + '\\' + imagename,'r')
    #excel face_bounding_box(top,bottom,left,right)
    #output/crop face_bounding_box(left,top,right,bottom)
    #img.resize((width, height))
    #for F1:
    delta_x = math.ceil(0.05*(test_table.cell(i+1,4).value - test_table.cell(i+1,3).value))
    delta_y = math.ceil(0.05*(test_table.cell(i+1,2).value - test_table.cell(i+1,1).value))
    faceboundingbox = [test_table.cell(i+1,3).value - delta_x,test_table.cell(i+1,1).value - delta_y,
                   test_table.cell(i+1,4).value + delta_y,test_table.cell(i+1,2).value + delta_y]
    region = img.crop(faceboundingbox)   #face bounding box image
    ## region_grey = region.convert('L')   #convert to grey image
    ## region_grey.save(IMAGESAVEURL_box_lfw_test + '\\' + true_imagename)
    region_resize = region.resize((39,39), Image.ANTIALIAS)   #F1 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_F1_lfw_test + '\\' + true_imagename)
    #for EN1:
    delta_EN1top = math.ceil(0.04*(test_table.cell(i+1,2).value - test_table.cell(i+1,1).value))
    delta_EN1bottom = math.ceil(0.84*(test_table.cell(i+1,2).value - test_table.cell(i+1,1).value))
    faceboundingbox_EN1 = [(test_table.cell(i+1,3).value - delta_x), (test_table.cell(i+1,1).value - delta_EN1top),
                   (test_table.cell(i+1,4).value + delta_x), (test_table.cell(i+1,1).value + delta_EN1bottom)]
    region_EN1 = img.crop(faceboundingbox_EN1)
    region_EN1_resize = region_EN1.resize((39,31), Image.ANTIALIAS)   #EN1 image
    region_EN1_resize_grey = region_EN1_resize.convert('L')   #convert to grey image
    region_EN1_resize_grey.save(IMAGESAVEURL_EN1_lfw_test + '\\' + true_imagename)
    #for NM1:
    delta_NM1top = math.ceil(0.18*(test_table.cell(i+1,2).value - test_table.cell(i+1,1).value))
    delta_NM1bottom = math.ceil(1.05*(test_table.cell(i+1,2).value - test_table.cell(i+1,1).value))
    faceboundingbox_NM1 = [(test_table.cell(i+1,3).value - delta_x), (test_table.cell(i+1,1).value + delta_NM1top),
                   (test_table.cell(i+1,4).value + delta_x), (test_table.cell(i+1,1).value + delta_NM1bottom)]
    region_NM1 = img.crop(faceboundingbox_NM1)
    region_NM1_resize = region_NM1.resize((39,31), Image.ANTIALIAS)   #EN1 image
    region_NM1_resize_grey = region_NM1_resize.convert('L')   #convert to grey image
    region_NM1_resize_grey.save(IMAGESAVEURL_NM1_lfw_test + '\\' + true_imagename)
'''

#this is for net_7876 test
'''
for i in range(2027):
    imagename = test_table.cell(i+1440,0).value
    true_imagename = imagename[9:]
    img = Image.open(IMAGEURL + '\\' + imagename,'r')
    #excel face_bounding_box(top,bottom,left,right)
    #output/crop face_bounding_box(left,top,right,bottom)
    #img.resize((width, height))
    #for F1:
    delta_x = math.ceil(0.05*(test_table.cell(i+1440,4).value - test_table.cell(i+1440,3).value))
    delta_y = math.ceil(0.05*(test_table.cell(i+1440,2).value - test_table.cell(i+1440,1).value))
    faceboundingbox = [test_table.cell(i+1440,3).value - delta_x, test_table.cell(i+1440,1).value - delta_y,
                   test_table.cell(i+1440,4).value + delta_x, test_table.cell(i+1440,2).value + delta_y]
    region = img.crop(faceboundingbox)   #face bounding box image
    ## region_grey = region.convert('L')   #convert to grey image
    ## region_grey.save(IMAGESAVEURL_box_net_test + '\\' + true_imagename)
    region_resize = region.resize((39,39), Image.ANTIALIAS)   #F1 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_F1_net_test + '\\' + true_imagename)
    #for EN1:
    delta_EN1top = math.ceil(0.04*(test_table.cell(i+1440,2).value - test_table.cell(i+1440,1).value))
    delta_EN1bottom = math.ceil(0.84*(test_table.cell(i+1440,2).value - test_table.cell(i+1440,1).value))
    faceboundingbox_EN1 = [(test_table.cell(i+1440,3).value - delta_x), (test_table.cell(i+1440,1).value - delta_EN1top),
                   (test_table.cell(i+1440,4).value + delta_x), (test_table.cell(i+1440,1).value + delta_EN1bottom)]
    region_EN1 = img.crop(faceboundingbox_EN1)
    region_EN1_resize = region_EN1.resize((39,31), Image.ANTIALIAS)   #EN1 image
    region_EN1_resize_grey = region_EN1_resize.convert('L')   #convert to grey image
    region_EN1_resize_grey.save(IMAGESAVEURL_EN1_net_test + '\\' + true_imagename)
    #for NM1:
    delta_NM1top = math.ceil(0.18*(test_table.cell(i+1440,2).value - test_table.cell(i+1440,1).value))
    delta_NM1bottom = math.ceil(1.05*(test_table.cell(i+1440,2).value - test_table.cell(i+1440,1).value))
    faceboundingbox_NM1 = [(test_table.cell(i+1440,3).value - delta_x), (test_table.cell(i+1440,1).value + delta_NM1top),
                   (test_table.cell(i+1440,4).value + delta_x), (test_table.cell(i+1440,1).value + delta_NM1bottom)]
    region_NM1 = img.crop(faceboundingbox_NM1)
    region_NM1_resize = region_NM1.resize((39,31), Image.ANTIALIAS)   #EN1 image
    region_NM1_resize_grey = region_NM1_resize.convert('L')   #convert to grey image
    region_NM1_resize_grey.save(IMAGESAVEURL_NM1_net_test + '\\' + true_imagename)
'''

