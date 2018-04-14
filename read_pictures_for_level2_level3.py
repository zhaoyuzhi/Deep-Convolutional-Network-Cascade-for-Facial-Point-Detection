# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 23:13:42 2018

@author: zhaoyuzhi
"""

from PIL import Image
import math
import xlrd

#read LFPW images according to excel file
IMAGEURL = "C:\\Users\\zhaoyuzhi\\Desktop\\train"

IMAGESAVEURL_LE21_lfw = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_LE21"
IMAGESAVEURL_LE22_lfw = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_LE22"
IMAGESAVEURL_RE21_lfw = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_RE21"
IMAGESAVEURL_RE22_lfw = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_RE22"
IMAGESAVEURL_N21_lfw = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_N21"
IMAGESAVEURL_N22_lfw = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_N22"
IMAGESAVEURL_LM21_lfw = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_LM21"
IMAGESAVEURL_LM22_lfw = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_LM22"
IMAGESAVEURL_RM21_lfw = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_RM21"
IMAGESAVEURL_RM22_lfw = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_RM22"
IMAGESAVEURL_LE31_lfw = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_LE31"
IMAGESAVEURL_LE32_lfw = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_LE32"
IMAGESAVEURL_RE31_lfw = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_RE31"
IMAGESAVEURL_RE32_lfw = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_RE32"
IMAGESAVEURL_N31_lfw = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_N31"
IMAGESAVEURL_N32_lfw = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_N32"
IMAGESAVEURL_LM31_lfw = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_LM31"
IMAGESAVEURL_LM32_lfw = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_LM32"
IMAGESAVEURL_RM31_lfw = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_RM31"
IMAGESAVEURL_RM32_lfw = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_RM32"

IMAGESAVEURL_LE21_net = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_LE21"
IMAGESAVEURL_LE22_net = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_LE22"
IMAGESAVEURL_RE21_net = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_RE21"
IMAGESAVEURL_RE22_net = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_RE22"
IMAGESAVEURL_N21_net = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_N21"
IMAGESAVEURL_N22_net = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_N22"
IMAGESAVEURL_LM21_net = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_LM21"
IMAGESAVEURL_LM22_net = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_LM22"
IMAGESAVEURL_RM21_net = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_RM21"
IMAGESAVEURL_RM22_net = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_RM22"
IMAGESAVEURL_LE31_net = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_LE31"
IMAGESAVEURL_LE32_net = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_LE32"
IMAGESAVEURL_RE31_net = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_RE31"
IMAGESAVEURL_RE32_net = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_RE32"
IMAGESAVEURL_N31_net = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_N31"
IMAGESAVEURL_N32_net = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_N32"
IMAGESAVEURL_LM31_net = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_LM31"
IMAGESAVEURL_LM32_net = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_LM32"
IMAGESAVEURL_RM31_net = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_RM31"
IMAGESAVEURL_RM32_net = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_RM32"

IMAGESAVEURL_LE21_lfw_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_LE21_test"
IMAGESAVEURL_LE22_lfw_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_LE22_test"
IMAGESAVEURL_RE21_lfw_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_RE21_test"
IMAGESAVEURL_RE22_lfw_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_RE22_test"
IMAGESAVEURL_N21_lfw_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_N21_test"
IMAGESAVEURL_N22_lfw_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_N22_test"
IMAGESAVEURL_LM21_lfw_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_LM21_test"
IMAGESAVEURL_LM22_lfw_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_LM22_test"
IMAGESAVEURL_RM21_lfw_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_RM21_test"
IMAGESAVEURL_RM22_lfw_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_RM22_test"
IMAGESAVEURL_LE31_lfw_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_LE31_test"
IMAGESAVEURL_LE32_lfw_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_LE32_test"
IMAGESAVEURL_RE31_lfw_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_RE31_test"
IMAGESAVEURL_RE32_lfw_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_RE32_test"
IMAGESAVEURL_N31_lfw_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_N31_test"
IMAGESAVEURL_N32_lfw_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_N32_test"
IMAGESAVEURL_LM31_lfw_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_LM31_test"
IMAGESAVEURL_LM32_lfw_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_LM32_test"
IMAGESAVEURL_RM31_lfw_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_RM31_test"
IMAGESAVEURL_RM32_lfw_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\lfw_5590_RM32_test"

IMAGESAVEURL_LE21_net_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_LE21_test"
IMAGESAVEURL_LE22_net_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_LE22_test"
IMAGESAVEURL_RE21_net_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_RE21_test"
IMAGESAVEURL_RE22_net_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_RE22_test"
IMAGESAVEURL_N21_net_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_N21_test"
IMAGESAVEURL_N22_net_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_N22_test"
IMAGESAVEURL_LM21_net_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_LM21_test"
IMAGESAVEURL_LM22_net_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_LM22_test"
IMAGESAVEURL_RM21_net_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_RM21_test"
IMAGESAVEURL_RM22_net_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_RM22_test"
IMAGESAVEURL_LE31_net_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_LE31_test"
IMAGESAVEURL_LE32_net_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_LE32_test"
IMAGESAVEURL_RE31_net_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_RE31_test"
IMAGESAVEURL_RE32_net_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_RE32_test"
IMAGESAVEURL_N31_net_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_N31_test"
IMAGESAVEURL_N32_net_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_N32_test"
IMAGESAVEURL_LM31_net_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_LM31_test"
IMAGESAVEURL_LM32_net_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_LM32_test"
IMAGESAVEURL_RM31_net_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_RM31_test"
IMAGESAVEURL_RM32_net_test = "C:\\Users\\zhaoyuzhi\\Desktop\\train\\net_7876_RM32_test"

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
    #for LE21:
    delta_x = math.ceil(0.16*(table.cell(i+1,4).value - table.cell(i+1,3).value))
    delta_y = math.ceil(0.16*(table.cell(i+1,2).value - table.cell(i+1,1).value))
    faceboundingbox = [(table.cell(i+1,5).value - delta_x), (table.cell(i+1,6).value - delta_y),
                   (table.cell(i+1,5).value + delta_x), (table.cell(i+1,6).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LE21 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LE21_lfw + '\\' + true_imagename)
    #for LE31:
    delta_x = math.ceil(0.11*(table.cell(i+1,4).value - table.cell(i+1,3).value))
    delta_y = math.ceil(0.11*(table.cell(i+1,2).value - table.cell(i+1,1).value))
    faceboundingbox = [(table.cell(i+1,5).value - delta_x), (table.cell(i+1,6).value - delta_y),
                   (table.cell(i+1,5).value + delta_x), (table.cell(i+1,6).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LE31 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LE31_lfw + '\\' + true_imagename)
    #for LE22:
    delta_x = math.ceil(0.18*(table.cell(i+1,4).value - table.cell(i+1,3).value))
    delta_y = math.ceil(0.18*(table.cell(i+1,2).value - table.cell(i+1,1).value))
    faceboundingbox = [(table.cell(i+1,5).value - delta_x), (table.cell(i+1,6).value - delta_y),
                   (table.cell(i+1,5).value + delta_x), (table.cell(i+1,6).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LE22 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LE22_lfw + '\\' + true_imagename)
    #for LE32:
    delta_x = math.ceil(0.12*(table.cell(i+1,4).value - table.cell(i+1,3).value))
    delta_y = math.ceil(0.12*(table.cell(i+1,2).value - table.cell(i+1,1).value))
    faceboundingbox = [(table.cell(i+1,5).value - delta_x), (table.cell(i+1,6).value - delta_y),
                   (table.cell(i+1,5).value + delta_x), (table.cell(i+1,6).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LE32 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LE32_lfw + '\\' + true_imagename)
    #for RE21:
    delta_x = math.ceil(0.16*(table.cell(i+1,4).value - table.cell(i+1,3).value))
    delta_y = math.ceil(0.16*(table.cell(i+1,2).value - table.cell(i+1,1).value))
    faceboundingbox = [(table.cell(i+1,7).value - delta_x), (table.cell(i+1,8).value - delta_y),
                   (table.cell(i+1,7).value + delta_x), (table.cell(i+1,8).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RE21 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RE21_lfw + '\\' + true_imagename)
    #for RE31:
    delta_x = math.ceil(0.11*(table.cell(i+1,4).value - table.cell(i+1,3).value))
    delta_y = math.ceil(0.11*(table.cell(i+1,2).value - table.cell(i+1,1).value))
    faceboundingbox = [(table.cell(i+1,7).value - delta_x), (table.cell(i+1,8).value - delta_y),
                   (table.cell(i+1,7).value + delta_x), (table.cell(i+1,8).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RE31 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RE31_lfw + '\\' + true_imagename)
    #for RE22:
    delta_x = math.ceil(0.18*(table.cell(i+1,4).value - table.cell(i+1,3).value))
    delta_y = math.ceil(0.18*(table.cell(i+1,2).value - table.cell(i+1,1).value))
    faceboundingbox = [(table.cell(i+1,7).value - delta_x), (table.cell(i+1,8).value - delta_y),
                   (table.cell(i+1,7).value + delta_x), (table.cell(i+1,8).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RE22 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RE22_lfw + '\\' + true_imagename)
    #for RE32:
    delta_x = math.ceil(0.12*(table.cell(i+1,4).value - table.cell(i+1,3).value))
    delta_y = math.ceil(0.12*(table.cell(i+1,2).value - table.cell(i+1,1).value))
    faceboundingbox = [(table.cell(i+1,7).value - delta_x), (table.cell(i+1,8).value - delta_y),
                   (table.cell(i+1,7).value + delta_x), (table.cell(i+1,8).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RE32 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RE32_lfw + '\\' + true_imagename)
    #for N21:
    delta_x = math.ceil(0.16*(table.cell(i+1,4).value - table.cell(i+1,3).value))
    delta_y = math.ceil(0.16*(table.cell(i+1,2).value - table.cell(i+1,1).value))
    faceboundingbox = [(table.cell(i+1,9).value - delta_x), (table.cell(i+1,10).value - delta_y),
                   (table.cell(i+1,9).value + delta_x), (table.cell(i+1,10).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #N21 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_N21_lfw + '\\' + true_imagename)
    #for N31:
    delta_x = math.ceil(0.11*(table.cell(i+1,4).value - table.cell(i+1,3).value))
    delta_y = math.ceil(0.11*(table.cell(i+1,2).value - table.cell(i+1,1).value))
    faceboundingbox = [(table.cell(i+1,9).value - delta_x), (table.cell(i+1,10).value - delta_y),
                   (table.cell(i+1,9).value + delta_x), (table.cell(i+1,10).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #N31 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_N31_lfw + '\\' + true_imagename)
    #for N22:
    delta_x = math.ceil(0.18*(table.cell(i+1,4).value - table.cell(i+1,3).value))
    delta_y = math.ceil(0.18*(table.cell(i+1,2).value - table.cell(i+1,1).value))
    faceboundingbox = [(table.cell(i+1,9).value - delta_x), (table.cell(i+1,10).value - delta_y),
                   (table.cell(i+1,9).value + delta_x), (table.cell(i+1,10).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #N22 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_N22_lfw + '\\' + true_imagename)
    #for N32:
    delta_x = math.ceil(0.12*(table.cell(i+1,4).value - table.cell(i+1,3).value))
    delta_y = math.ceil(0.12*(table.cell(i+1,2).value - table.cell(i+1,1).value))
    faceboundingbox = [(table.cell(i+1,9).value - delta_x), (table.cell(i+1,10).value - delta_y),
                   (table.cell(i+1,9).value + delta_x), (table.cell(i+1,10).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #N32 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_N32_lfw + '\\' + true_imagename)
    #for LM21:
    delta_x = math.ceil(0.16*(table.cell(i+1,4).value - table.cell(i+1,3).value))
    delta_y = math.ceil(0.16*(table.cell(i+1,2).value - table.cell(i+1,1).value))
    faceboundingbox = [(table.cell(i+1,11).value - delta_x), (table.cell(i+1,12).value - delta_y),
                   (table.cell(i+1,11).value + delta_x), (table.cell(i+1,12).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LM21 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LM21_lfw + '\\' + true_imagename)
    #for LM31:
    delta_x = math.ceil(0.11*(table.cell(i+1,4).value - table.cell(i+1,3).value))
    delta_y = math.ceil(0.11*(table.cell(i+1,2).value - table.cell(i+1,1).value))
    faceboundingbox = [(table.cell(i+1,11).value - delta_x), (table.cell(i+1,12).value - delta_y),
                   (table.cell(i+1,11).value + delta_x), (table.cell(i+1,12).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LM31 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LM31_lfw + '\\' + true_imagename)
    #for LM22:
    delta_x = math.ceil(0.18*(table.cell(i+1,4).value - table.cell(i+1,3).value))
    delta_y = math.ceil(0.18*(table.cell(i+1,2).value - table.cell(i+1,1).value))
    faceboundingbox = [(table.cell(i+1,11).value - delta_x), (table.cell(i+1,12).value - delta_y),
                   (table.cell(i+1,11).value + delta_x), (table.cell(i+1,12).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LM22 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LM22_lfw + '\\' + true_imagename)
    #for LM32:
    delta_x = math.ceil(0.12*(table.cell(i+1,4).value - table.cell(i+1,3).value))
    delta_y = math.ceil(0.12*(table.cell(i+1,2).value - table.cell(i+1,1).value))
    faceboundingbox = [(table.cell(i+1,11).value - delta_x), (table.cell(i+1,12).value - delta_y),
                   (table.cell(i+1,11).value + delta_x), (table.cell(i+1,12).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LM32 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LM32_lfw + '\\' + true_imagename)
    #for RM21:
    delta_x = math.ceil(0.16*(table.cell(i+1,4).value - table.cell(i+1,3).value))
    delta_y = math.ceil(0.16*(table.cell(i+1,2).value - table.cell(i+1,1).value))
    faceboundingbox = [(table.cell(i+1,13).value - delta_x), (table.cell(i+1,14).value - delta_y),
                   (table.cell(i+1,13).value + delta_x), (table.cell(i+1,14).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RM21 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RM21_lfw + '\\' + true_imagename)
    #for RM31:
    delta_x = math.ceil(0.11*(table.cell(i+1,4).value - table.cell(i+1,3).value))
    delta_y = math.ceil(0.11*(table.cell(i+1,2).value - table.cell(i+1,1).value))
    faceboundingbox = [(table.cell(i+1,13).value - delta_x), (table.cell(i+1,14).value - delta_y),
                   (table.cell(i+1,13).value + delta_x), (table.cell(i+1,14).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RM31 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RM31_lfw + '\\' + true_imagename)
    #for RM22:
    delta_x = math.ceil(0.18*(table.cell(i+1,4).value - table.cell(i+1,3).value))
    delta_y = math.ceil(0.18*(table.cell(i+1,2).value - table.cell(i+1,1).value))
    faceboundingbox = [(table.cell(i+1,13).value - delta_x), (table.cell(i+1,14).value - delta_y),
                   (table.cell(i+1,13).value + delta_x), (table.cell(i+1,14).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RM22 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RM22_lfw + '\\' + true_imagename)
    #for RM32:
    delta_x = math.ceil(0.12*(table.cell(i+1,4).value - table.cell(i+1,3).value))
    delta_y = math.ceil(0.12*(table.cell(i+1,2).value - table.cell(i+1,1).value))
    faceboundingbox = [(table.cell(i+1,13).value - delta_x), (table.cell(i+1,14).value - delta_y),
                   (table.cell(i+1,13).value + delta_x), (table.cell(i+1,14).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RM32 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RM32_lfw + '\\' + true_imagename)
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
    #for LE21:
    delta_x = math.ceil(0.16*(table.cell(i+4152,4).value - table.cell(i+4152,3).value))
    delta_y = math.ceil(0.16*(table.cell(i+4152,2).value - table.cell(i+4152,1).value))
    faceboundingbox = [(table.cell(i+4152,5).value - delta_x), (table.cell(i+4152,6).value - delta_y),
                   (table.cell(i+4152,5).value + delta_x), (table.cell(i+4152,6).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LE21 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LE21_net + '\\' + true_imagename)
    #for LE31:
    delta_x = math.ceil(0.11*(table.cell(i+4152,4).value - table.cell(i+4152,3).value))
    delta_y = math.ceil(0.11*(table.cell(i+4152,2).value - table.cell(i+4152,1).value))
    faceboundingbox = [(table.cell(i+4152,5).value - delta_x), (table.cell(i+4152,6).value - delta_y),
                   (table.cell(i+4152,5).value + delta_x), (table.cell(i+4152,6).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LE31 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LE31_net + '\\' + true_imagename)
    #for LE22:
    delta_x = math.ceil(0.18*(table.cell(i+4152,4).value - table.cell(i+4152,3).value))
    delta_y = math.ceil(0.18*(table.cell(i+4152,2).value - table.cell(i+4152,1).value))
    faceboundingbox = [(table.cell(i+4152,5).value - delta_x), (table.cell(i+4152,6).value - delta_y),
                   (table.cell(i+4152,5).value + delta_x), (table.cell(i+4152,6).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LE22 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LE22_net + '\\' + true_imagename)
    #for LE32:
    delta_x = math.ceil(0.12*(table.cell(i+4152,4).value - table.cell(i+4152,3).value))
    delta_y = math.ceil(0.12*(table.cell(i+4152,2).value - table.cell(i+4152,1).value))
    faceboundingbox = [(table.cell(i+4152,5).value - delta_x), (table.cell(i+4152,6).value - delta_y),
                   (table.cell(i+4152,5).value + delta_x), (table.cell(i+4152,6).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LE32 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LE32_net + '\\' + true_imagename)
    #for RE21:
    delta_x = math.ceil(0.16*(table.cell(i+4152,4).value - table.cell(i+4152,3).value))
    delta_y = math.ceil(0.16*(table.cell(i+4152,2).value - table.cell(i+4152,1).value))
    faceboundingbox = [(table.cell(i+4152,7).value - delta_x), (table.cell(i+4152,8).value - delta_y),
                   (table.cell(i+4152,7).value + delta_x), (table.cell(i+4152,8).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RE21 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RE21_net + '\\' + true_imagename)
    #for RE31:
    delta_x = math.ceil(0.11*(table.cell(i+4152,4).value - table.cell(i+4152,3).value))
    delta_y = math.ceil(0.11*(table.cell(i+4152,2).value - table.cell(i+4152,1).value))
    faceboundingbox = [(table.cell(i+4152,7).value - delta_x), (table.cell(i+4152,8).value - delta_y),
                   (table.cell(i+4152,7).value + delta_x), (table.cell(i+4152,8).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RE31 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RE31_net + '\\' + true_imagename)
    #for RE22:
    delta_x = math.ceil(0.18*(table.cell(i+4152,4).value - table.cell(i+4152,3).value))
    delta_y = math.ceil(0.18*(table.cell(i+4152,2).value - table.cell(i+4152,1).value))
    faceboundingbox = [(table.cell(i+4152,7).value - delta_x), (table.cell(i+4152,8).value - delta_y),
                   (table.cell(i+4152,7).value + delta_x), (table.cell(i+4152,8).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RE22 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RE22_net + '\\' + true_imagename)
    #for RE32:
    delta_x = math.ceil(0.12*(table.cell(i+4152,4).value - table.cell(i+4152,3).value))
    delta_y = math.ceil(0.12*(table.cell(i+4152,2).value - table.cell(i+4152,1).value))
    faceboundingbox = [(table.cell(i+4152,7).value - delta_x), (table.cell(i+4152,8).value - delta_y),
                   (table.cell(i+4152,7).value + delta_x), (table.cell(i+4152,8).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RE32 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RE32_net + '\\' + true_imagename)
    #for N21:
    delta_x = math.ceil(0.16*(table.cell(i+4152,4).value - table.cell(i+4152,3).value))
    delta_y = math.ceil(0.16*(table.cell(i+4152,2).value - table.cell(i+4152,1).value))
    faceboundingbox = [(table.cell(i+4152,9).value - delta_x), (table.cell(i+4152,10).value - delta_y),
                   (table.cell(i+4152,9).value + delta_x), (table.cell(i+4152,10).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #N21 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_N21_net + '\\' + true_imagename)
    #for N31:
    delta_x = math.ceil(0.11*(table.cell(i+4152,4).value - table.cell(i+4152,3).value))
    delta_y = math.ceil(0.11*(table.cell(i+4152,2).value - table.cell(i+4152,1).value))
    faceboundingbox = [(table.cell(i+4152,9).value - delta_x), (table.cell(i+4152,10).value - delta_y),
                   (table.cell(i+4152,9).value + delta_x), (table.cell(i+4152,10).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #N31 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_N31_net + '\\' + true_imagename)
    #for N22:
    delta_x = math.ceil(0.18*(table.cell(i+4152,4).value - table.cell(i+4152,3).value))
    delta_y = math.ceil(0.18*(table.cell(i+4152,2).value - table.cell(i+4152,1).value))
    faceboundingbox = [(table.cell(i+4152,9).value - delta_x), (table.cell(i+4152,10).value - delta_y),
                   (table.cell(i+4152,9).value + delta_x), (table.cell(i+4152,10).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #N22 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_N22_net+ '\\' + true_imagename)
    #for N32:
    delta_x = math.ceil(0.12*(table.cell(i+4152,4).value - table.cell(i+4152,3).value))
    delta_y = math.ceil(0.12*(table.cell(i+4152,2).value - table.cell(i+4152,1).value))
    faceboundingbox = [(table.cell(i+4152,9).value - delta_x), (table.cell(i+4152,10).value - delta_y),
                   (table.cell(i+4152,9).value + delta_x), (table.cell(i+4152,10).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #N32 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_N32_net + '\\' + true_imagename)
    #for LM21:
    delta_x = math.ceil(0.16*(table.cell(i+4152,4).value - table.cell(i+4152,3).value))
    delta_y = math.ceil(0.16*(table.cell(i+4152,2).value - table.cell(i+4152,1).value))
    faceboundingbox = [(table.cell(i+4152,11).value - delta_x), (table.cell(i+4152,12).value - delta_y),
                   (table.cell(i+4152,11).value + delta_x), (table.cell(i+4152,12).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LM21 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LM21_net + '\\' + true_imagename)
    #for LM31:
    delta_x = math.ceil(0.11*(table.cell(i+4152,4).value - table.cell(i+4152,3).value))
    delta_y = math.ceil(0.11*(table.cell(i+4152,2).value - table.cell(i+4152,1).value))
    faceboundingbox = [(table.cell(i+4152,11).value - delta_x), (table.cell(i+4152,12).value - delta_y),
                   (table.cell(i+4152,11).value + delta_x), (table.cell(i+4152,12).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LM31 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LM31_net + '\\' + true_imagename)
    #for LM22:
    delta_x = math.ceil(0.18*(table.cell(i+4152,4).value - table.cell(i+4152,3).value))
    delta_y = math.ceil(0.18*(table.cell(i+4152,2).value - table.cell(i+4152,1).value))
    faceboundingbox = [(table.cell(i+4152,11).value - delta_x), (table.cell(i+4152,12).value - delta_y),
                   (table.cell(i+4152,11).value + delta_x), (table.cell(i+4152,12).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LM22 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LM22_net + '\\' + true_imagename)
    #for LM32:
    delta_x = math.ceil(0.12*(table.cell(i+4152,4).value - table.cell(i+4152,3).value))
    delta_y = math.ceil(0.12*(table.cell(i+4152,2).value - table.cell(i+4152,1).value))
    faceboundingbox = [(table.cell(i+4152,11).value - delta_x), (table.cell(i+4152,12).value - delta_y),
                   (table.cell(i+4152,11).value + delta_x), (table.cell(i+4152,12).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LM32 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LM32_net + '\\' + true_imagename)
    #for RM21:
    delta_x = math.ceil(0.16*(table.cell(i+4152,4).value - table.cell(i+4152,3).value))
    delta_y = math.ceil(0.16*(table.cell(i+4152,2).value - table.cell(i+4152,1).value))
    faceboundingbox = [(table.cell(i+4152,13).value - delta_x), (table.cell(i+4152,14).value - delta_y),
                   (table.cell(i+4152,13).value + delta_x), (table.cell(i+4152,14).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RM21 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RM21_net + '\\' + true_imagename)
    #for RM31:
    delta_x = math.ceil(0.11*(table.cell(i+4152,4).value - table.cell(i+4152,3).value))
    delta_y = math.ceil(0.11*(table.cell(i+4152,2).value - table.cell(i+4152,1).value))
    faceboundingbox = [(table.cell(i+4152,13).value - delta_x), (table.cell(i+4152,14).value - delta_y),
                   (table.cell(i+4152,13).value + delta_x), (table.cell(i+4152,14).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RM31 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RM31_net + '\\' + true_imagename)
    #for RM22:
    delta_x = math.ceil(0.18*(table.cell(i+4152,4).value - table.cell(i+4152,3).value))
    delta_y = math.ceil(0.18*(table.cell(i+4152,2).value - table.cell(i+4152,1).value))
    faceboundingbox = [(table.cell(i+4152,13).value - delta_x), (table.cell(i+4152,14).value - delta_y),
                   (table.cell(i+4152,13).value + delta_x), (table.cell(i+4152,14).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RM22 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RM22_net + '\\' + true_imagename)
    #for RM32:
    delta_x = math.ceil(0.12*(table.cell(i+4152,4).value - table.cell(i+4152,3).value))
    delta_y = math.ceil(0.12*(table.cell(i+4152,2).value - table.cell(i+4152,1).value))
    faceboundingbox = [(table.cell(i+4152,13).value - delta_x), (table.cell(i+4152,14).value - delta_y),
                   (table.cell(i+4152,13).value + delta_x), (table.cell(i+4152,14).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RM32 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RM32_net + '\\' + true_imagename)
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
    #for LE21:
    delta_x = math.ceil(0.16*(test_table.cell(i+1,4).value - test_table.cell(i+1,3).value))
    delta_y = math.ceil(0.16*(test_table.cell(i+1,2).value - test_table.cell(i+1,1).value))
    faceboundingbox = [(test_table.cell(i+1,5).value - delta_x), (test_table.cell(i+1,6).value - delta_y),
                   (test_table.cell(i+1,5).value + delta_x), (test_table.cell(i+1,6).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LE21 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LE21_lfw_test + '\\' + true_imagename)
    #for LE31:
    delta_x = math.ceil(0.11*(test_table.cell(i+1,4).value - test_table.cell(i+1,3).value))
    delta_y = math.ceil(0.11*(test_table.cell(i+1,2).value - test_table.cell(i+1,1).value))
    faceboundingbox = [(test_table.cell(i+1,5).value - delta_x), (test_table.cell(i+1,6).value - delta_y),
                   (test_table.cell(i+1,5).value + delta_x), (test_table.cell(i+1,6).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LE31 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LE31_lfw_test + '\\' + true_imagename)
    #for LE22:
    delta_x = math.ceil(0.18*(test_table.cell(i+1,4).value - test_table.cell(i+1,3).value))
    delta_y = math.ceil(0.18*(test_table.cell(i+1,2).value - test_table.cell(i+1,1).value))
    faceboundingbox = [(test_table.cell(i+1,5).value - delta_x), (test_table.cell(i+1,6).value - delta_y),
                   (test_table.cell(i+1,5).value + delta_x), (test_table.cell(i+1,6).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LE22 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LE22_lfw_test + '\\' + true_imagename)
    #for LE32:
    delta_x = math.ceil(0.12*(test_table.cell(i+1,4).value - test_table.cell(i+1,3).value))
    delta_y = math.ceil(0.12*(test_table.cell(i+1,2).value - test_table.cell(i+1,1).value))
    faceboundingbox = [(test_table.cell(i+1,5).value - delta_x), (test_table.cell(i+1,6).value - delta_y),
                   (test_table.cell(i+1,5).value + delta_x), (test_table.cell(i+1,6).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LE32 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LE32_lfw_test+ '\\' + true_imagename)
    #for RE21:
    delta_x = math.ceil(0.16*(test_table.cell(i+1,4).value - test_table.cell(i+1,3).value))
    delta_y = math.ceil(0.16*(test_table.cell(i+1,2).value - test_table.cell(i+1,1).value))
    faceboundingbox = [(test_table.cell(i+1,7).value - delta_x), (test_table.cell(i+1,8).value - delta_y),
                   (test_table.cell(i+1,7).value + delta_x), (test_table.cell(i+1,8).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RE21 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RE21_lfw_test + '\\' + true_imagename)
    #for RE31:
    delta_x = math.ceil(0.11*(test_table.cell(i+1,4).value - test_table.cell(i+1,3).value))
    delta_y = math.ceil(0.11*(test_table.cell(i+1,2).value - test_table.cell(i+1,1).value))
    faceboundingbox = [(test_table.cell(i+1,7).value - delta_x), (test_table.cell(i+1,8).value - delta_y),
                   (test_table.cell(i+1,7).value + delta_x), (test_table.cell(i+1,8).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RE31 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RE31_lfw_test + '\\' + true_imagename)
    #for RE22:
    delta_x = math.ceil(0.18*(test_table.cell(i+1,4).value - test_table.cell(i+1,3).value))
    delta_y = math.ceil(0.18*(test_table.cell(i+1,2).value - test_table.cell(i+1,1).value))
    faceboundingbox = [(test_table.cell(i+1,7).value - delta_x), (test_table.cell(i+1,8).value - delta_y),
                   (test_table.cell(i+1,7).value + delta_x), (test_table.cell(i+1,8).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RE22 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RE22_lfw_test + '\\' + true_imagename)
    #for RE32:
    delta_x = math.ceil(0.12*(test_table.cell(i+1,4).value - test_table.cell(i+1,3).value))
    delta_y = math.ceil(0.12*(test_table.cell(i+1,2).value - test_table.cell(i+1,1).value))
    faceboundingbox = [(test_table.cell(i+1,7).value - delta_x), (test_table.cell(i+1,8).value - delta_y),
                   (test_table.cell(i+1,7).value + delta_x), (test_table.cell(i+1,8).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RE32 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RE32_lfw_test + '\\' + true_imagename)
    #for N21:
    delta_x = math.ceil(0.16*(test_table.cell(i+1,4).value - test_table.cell(i+1,3).value))
    delta_y = math.ceil(0.16*(test_table.cell(i+1,2).value - test_table.cell(i+1,1).value))
    faceboundingbox = [(test_table.cell(i+1,9).value - delta_x), (test_table.cell(i+1,10).value - delta_y),
                   (test_table.cell(i+1,9).value + delta_x), (test_table.cell(i+1,10).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #N21 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_N21_lfw_test + '\\' + true_imagename)
    #for N31:
    delta_x = math.ceil(0.11*(test_table.cell(i+1,4).value - test_table.cell(i+1,3).value))
    delta_y = math.ceil(0.11*(test_table.cell(i+1,2).value - test_table.cell(i+1,1).value))
    faceboundingbox = [(test_table.cell(i+1,9).value - delta_x), (test_table.cell(i+1,10).value - delta_y),
                   (test_table.cell(i+1,9).value + delta_x), (test_table.cell(i+1,10).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #N31 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_N31_lfw_test + '\\' + true_imagename)
    #for N22:
    delta_x = math.ceil(0.18*(test_table.cell(i+1,4).value - test_table.cell(i+1,3).value))
    delta_y = math.ceil(0.18*(test_table.cell(i+1,2).value - test_table.cell(i+1,1).value))
    faceboundingbox = [(test_table.cell(i+1,9).value - delta_x), (test_table.cell(i+1,10).value - delta_y),
                   (test_table.cell(i+1,9).value + delta_x), (test_table.cell(i+1,10).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #N22 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_N22_lfw_test + '\\' + true_imagename)
    #for N32:
    delta_x = math.ceil(0.12*(test_table.cell(i+1,4).value - test_table.cell(i+1,3).value))
    delta_y = math.ceil(0.12*(test_table.cell(i+1,2).value - test_table.cell(i+1,1).value))
    faceboundingbox = [(test_table.cell(i+1,9).value - delta_x), (test_table.cell(i+1,10).value - delta_y),
                   (test_table.cell(i+1,9).value + delta_x), (test_table.cell(i+1,10).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #N32 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_N32_lfw_test + '\\' + true_imagename)
    #for LM21:
    delta_x = math.ceil(0.16*(test_table.cell(i+1,4).value - test_table.cell(i+1,3).value))
    delta_y = math.ceil(0.16*(test_table.cell(i+1,2).value - test_table.cell(i+1,1).value))
    faceboundingbox = [(test_table.cell(i+1,11).value - delta_x), (test_table.cell(i+1,12).value - delta_y),
                   (test_table.cell(i+1,11).value + delta_x), (test_table.cell(i+1,12).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LM21 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LM21_lfw_test + '\\' + true_imagename)
    #for LM31:
    delta_x = math.ceil(0.11*(test_table.cell(i+1,4).value - test_table.cell(i+1,3).value))
    delta_y = math.ceil(0.11*(test_table.cell(i+1,2).value - test_table.cell(i+1,1).value))
    faceboundingbox = [(test_table.cell(i+1,11).value - delta_x), (test_table.cell(i+1,12).value - delta_y),
                   (test_table.cell(i+1,11).value + delta_x), (test_table.cell(i+1,12).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LM31 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LM31_lfw_test + '\\' + true_imagename)
    #for LM22:
    delta_x = math.ceil(0.18*(test_table.cell(i+1,4).value - test_table.cell(i+1,3).value))
    delta_y = math.ceil(0.18*(test_table.cell(i+1,2).value - test_table.cell(i+1,1).value))
    faceboundingbox = [(test_table.cell(i+1,11).value - delta_x), (test_table.cell(i+1,12).value - delta_y),
                   (test_table.cell(i+1,11).value + delta_x), (test_table.cell(i+1,12).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LM22 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LM22_lfw_test + '\\' + true_imagename)
    #for LM32:
    delta_x = math.ceil(0.12*(test_table.cell(i+1,4).value - test_table.cell(i+1,3).value))
    delta_y = math.ceil(0.12*(test_table.cell(i+1,2).value - test_table.cell(i+1,1).value))
    faceboundingbox = [(test_table.cell(i+1,11).value - delta_x), (test_table.cell(i+1,12).value - delta_y),
                   (test_table.cell(i+1,11).value + delta_x), (test_table.cell(i+1,12).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LM32 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LM32_lfw_test + '\\' + true_imagename)
    #for RM21:
    delta_x = math.ceil(0.16*(test_table.cell(i+1,4).value - test_table.cell(i+1,3).value))
    delta_y = math.ceil(0.16*(test_table.cell(i+1,2).value - test_table.cell(i+1,1).value))
    faceboundingbox = [(test_table.cell(i+1,13).value - delta_x), (test_table.cell(i+1,14).value - delta_y),
                   (test_table.cell(i+1,13).value + delta_x), (test_table.cell(i+1,14).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RM21 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RM21_lfw_test + '\\' + true_imagename)
    #for RM31:
    delta_x = math.ceil(0.11*(test_table.cell(i+1,4).value - test_table.cell(i+1,3).value))
    delta_y = math.ceil(0.11*(test_table.cell(i+1,2).value - test_table.cell(i+1,1).value))
    faceboundingbox = [(test_table.cell(i+1,13).value - delta_x), (test_table.cell(i+1,14).value - delta_y),
                   (test_table.cell(i+1,13).value + delta_x), (test_table.cell(i+1,14).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RM31 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RM31_lfw_test + '\\' + true_imagename)
    #for RM22:
    delta_x = math.ceil(0.18*(test_table.cell(i+1,4).value - test_table.cell(i+1,3).value))
    delta_y = math.ceil(0.18*(test_table.cell(i+1,2).value - test_table.cell(i+1,1).value))
    faceboundingbox = [(test_table.cell(i+1,13).value - delta_x), (test_table.cell(i+1,14).value - delta_y),
                   (test_table.cell(i+1,13).value + delta_x), (test_table.cell(i+1,14).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RM22 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RM22_lfw_test + '\\' + true_imagename)
    #for RM32:
    delta_x = math.ceil(0.12*(test_table.cell(i+1,4).value - test_table.cell(i+1,3).value))
    delta_y = math.ceil(0.12*(test_table.cell(i+1,2).value - test_table.cell(i+1,1).value))
    faceboundingbox = [(test_table.cell(i+1,13).value - delta_x), (test_table.cell(i+1,14).value - delta_y),
                   (test_table.cell(i+1,13).value + delta_x), (test_table.cell(i+1,14).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RM32 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RM32_lfw_test + '\\' + true_imagename)
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
    #for LE21:
    delta_x = math.ceil(0.16*(test_table.cell(i+1440,4).value - test_table.cell(i+1440,3).value))
    delta_y = math.ceil(0.16*(test_table.cell(i+1440,2).value - test_table.cell(i+1440,1).value))
    faceboundingbox = [(test_table.cell(i+1440,5).value - delta_x), (test_table.cell(i+1440,6).value - delta_y),
                   (test_table.cell(i+1440,5).value + delta_x), (test_table.cell(i+1440,6).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LE21 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LE21_net_test + '\\' + true_imagename)
    #for LE31:
    delta_x = math.ceil(0.11*(test_table.cell(i+1440,4).value - test_table.cell(i+1440,3).value))
    delta_y = math.ceil(0.11*(test_table.cell(i+1440,2).value - test_table.cell(i+1440,1).value))
    faceboundingbox = [(test_table.cell(i+1440,5).value - delta_x), (test_table.cell(i+1440,6).value - delta_y),
                   (test_table.cell(i+1440,5).value + delta_x), (test_table.cell(i+1440,6).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LE31 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LE31_net_test + '\\' + true_imagename)
    #for LE22:
    delta_x = math.ceil(0.18*(test_table.cell(i+1440,4).value - test_table.cell(i+1440,3).value))
    delta_y = math.ceil(0.18*(test_table.cell(i+1440,2).value - test_table.cell(i+1440,1).value))
    faceboundingbox = [(test_table.cell(i+1440,5).value - delta_x), (test_table.cell(i+1440,6).value - delta_y),
                   (test_table.cell(i+1440,5).value + delta_x), (test_table.cell(i+1440,6).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LE22 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LE22_net_test + '\\' + true_imagename)
    #for LE32:
    delta_x = math.ceil(0.12*(test_table.cell(i+1440,4).value - test_table.cell(i+1440,3).value))
    delta_y = math.ceil(0.12*(test_table.cell(i+1440,2).value - test_table.cell(i+1440,1).value))
    faceboundingbox = [(test_table.cell(i+1440,5).value - delta_x), (test_table.cell(i+1440,6).value - delta_y),
                   (test_table.cell(i+1440,5).value + delta_x), (test_table.cell(i+1440,6).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LE32 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LE32_net_test+ '\\' + true_imagename)
    #for RE21:
    delta_x = math.ceil(0.16*(test_table.cell(i+1440,4).value - test_table.cell(i+1440,3).value))
    delta_y = math.ceil(0.16*(test_table.cell(i+1440,2).value - test_table.cell(i+1440,1).value))
    faceboundingbox = [(test_table.cell(i+1440,7).value - delta_x), (test_table.cell(i+1440,8).value - delta_y),
                   (test_table.cell(i+1440,7).value + delta_x), (test_table.cell(i+1440,8).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RE21 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RE21_net_test + '\\' + true_imagename)
    #for RE31:
    delta_x = math.ceil(0.11*(test_table.cell(i+1440,4).value - test_table.cell(i+1440,3).value))
    delta_y = math.ceil(0.11*(test_table.cell(i+1440,2).value - test_table.cell(i+1440,1).value))
    faceboundingbox = [(test_table.cell(i+1440,7).value - delta_x), (test_table.cell(i+1440,8).value - delta_y),
                   (test_table.cell(i+1440,7).value + delta_x), (test_table.cell(i+1440,8).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RE31 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RE31_net_test + '\\' + true_imagename)
    #for RE22:
    delta_x = math.ceil(0.18*(test_table.cell(i+1440,4).value - test_table.cell(i+1440,3).value))
    delta_y = math.ceil(0.18*(test_table.cell(i+1440,2).value - test_table.cell(i+1440,1).value))
    faceboundingbox = [(test_table.cell(i+1440,7).value - delta_x), (test_table.cell(i+1440,8).value - delta_y),
                   (test_table.cell(i+1440,7).value + delta_x), (test_table.cell(i+1440,8).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RE22 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RE22_net_test + '\\' + true_imagename)
    #for RE32:
    delta_x = math.ceil(0.12*(test_table.cell(i+1440,4).value - test_table.cell(i+1440,3).value))
    delta_y = math.ceil(0.12*(test_table.cell(i+1440,2).value - test_table.cell(i+1440,1).value))
    faceboundingbox = [(test_table.cell(i+1440,7).value - delta_x), (test_table.cell(i+1440,8).value - delta_y),
                   (test_table.cell(i+1440,7).value + delta_x), (test_table.cell(i+1440,8).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RE32 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RE32_net_test + '\\' + true_imagename)
    #for N21:
    delta_x = math.ceil(0.16*(test_table.cell(i+1440,4).value - test_table.cell(i+1440,3).value))
    delta_y = math.ceil(0.16*(test_table.cell(i+1440,2).value - test_table.cell(i+1440,1).value))
    faceboundingbox = [(test_table.cell(i+1440,9).value - delta_x), (test_table.cell(i+1440,10).value - delta_y),
                   (test_table.cell(i+1440,9).value + delta_x), (test_table.cell(i+1440,10).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #N21 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_N21_net_test + '\\' + true_imagename)
    #for N31:
    delta_x = math.ceil(0.11*(test_table.cell(i+1440,4).value - test_table.cell(i+1440,3).value))
    delta_y = math.ceil(0.11*(test_table.cell(i+1440,2).value - test_table.cell(i+1440,1).value))
    faceboundingbox = [(test_table.cell(i+1440,9).value - delta_x), (test_table.cell(i+1440,10).value - delta_y),
                   (test_table.cell(i+1440,9).value + delta_x), (test_table.cell(i+1440,10).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #N31 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_N31_net_test + '\\' + true_imagename)
    #for N22:
    delta_x = math.ceil(0.18*(test_table.cell(i+1440,4).value - test_table.cell(i+1440,3).value))
    delta_y = math.ceil(0.18*(test_table.cell(i+1440,2).value - test_table.cell(i+1440,1).value))
    faceboundingbox = [(test_table.cell(i+1440,9).value - delta_x), (test_table.cell(i+1440,10).value - delta_y),
                   (test_table.cell(i+1440,9).value + delta_x), (test_table.cell(i+1440,10).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #N22 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_N22_net_test + '\\' + true_imagename)
    #for N32:
    delta_x = math.ceil(0.12*(test_table.cell(i+1440,4).value - test_table.cell(i+1440,3).value))
    delta_y = math.ceil(0.12*(test_table.cell(i+1440,2).value - test_table.cell(i+1440,1).value))
    faceboundingbox = [(test_table.cell(i+1440,9).value - delta_x), (test_table.cell(i+1440,10).value - delta_y),
                   (test_table.cell(i+1440,9).value + delta_x), (test_table.cell(i+1440,10).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #N32 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_N32_net_test + '\\' + true_imagename)
    #for LM21:
    delta_x = math.ceil(0.16*(test_table.cell(i+1440,4).value - test_table.cell(i+1440,3).value))
    delta_y = math.ceil(0.16*(test_table.cell(i+1440,2).value - test_table.cell(i+1440,1).value))
    faceboundingbox = [(test_table.cell(i+1440,11).value - delta_x), (test_table.cell(i+1440,12).value - delta_y),
                   (test_table.cell(i+1440,11).value + delta_x), (test_table.cell(i+1440,12).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LM21 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LM21_net_test + '\\' + true_imagename)
    #for LM31:
    delta_x = math.ceil(0.11*(test_table.cell(i+1440,4).value - test_table.cell(i+1440,3).value))
    delta_y = math.ceil(0.11*(test_table.cell(i+1440,2).value - test_table.cell(i+1440,1).value))
    faceboundingbox = [(test_table.cell(i+1440,11).value - delta_x), (test_table.cell(i+1440,12).value - delta_y),
                   (test_table.cell(i+1440,11).value + delta_x), (test_table.cell(i+1440,12).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LM31 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LM31_net_test + '\\' + true_imagename)
    #for LM22:
    delta_x = math.ceil(0.18*(test_table.cell(i+1440,4).value - test_table.cell(i+1440,3).value))
    delta_y = math.ceil(0.18*(test_table.cell(i+1440,2).value - test_table.cell(i+1440,1).value))
    faceboundingbox = [(test_table.cell(i+1440,11).value - delta_x), (test_table.cell(i+1440,12).value - delta_y),
                   (test_table.cell(i+1440,11).value + delta_x), (test_table.cell(i+1440,12).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LM22 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LM22_net_test + '\\' + true_imagename)
    #for LM32:
    delta_x = math.ceil(0.12*(test_table.cell(i+1440,4).value - test_table.cell(i+1440,3).value))
    delta_y = math.ceil(0.12*(test_table.cell(i+1440,2).value - test_table.cell(i+1440,1).value))
    faceboundingbox = [(test_table.cell(i+1440,11).value - delta_x), (test_table.cell(i+1440,12).value - delta_y),
                   (test_table.cell(i+1440,11).value + delta_x), (test_table.cell(i+1440,12).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #LM32 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_LM32_net_test + '\\' + true_imagename)
    #for RM21:
    delta_x = math.ceil(0.16*(test_table.cell(i+1440,4).value - test_table.cell(i+1440,3).value))
    delta_y = math.ceil(0.16*(test_table.cell(i+1440,2).value - test_table.cell(i+1440,1).value))
    faceboundingbox = [(test_table.cell(i+1440,13).value - delta_x), (test_table.cell(i+1440,14).value - delta_y),
                   (test_table.cell(i+1440,13).value + delta_x), (test_table.cell(i+1440,14).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RM21 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RM21_net_test + '\\' + true_imagename)
    #for RM31:
    delta_x = math.ceil(0.11*(test_table.cell(i+1440,4).value - test_table.cell(i+1440,3).value))
    delta_y = math.ceil(0.11*(test_table.cell(i+1440,2).value - test_table.cell(i+1440,1).value))
    faceboundingbox = [(test_table.cell(i+1440,13).value - delta_x), (test_table.cell(i+1440,14).value - delta_y),
                   (test_table.cell(i+1440,13).value + delta_x), (test_table.cell(i+1440,14).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RM31 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RM31_net_test + '\\' + true_imagename)
    #for RM22:
    delta_x = math.ceil(0.18*(test_table.cell(i+1440,4).value - test_table.cell(i+1440,3).value))
    delta_y = math.ceil(0.18*(test_table.cell(i+1440,2).value - test_table.cell(i+1440,1).value))
    faceboundingbox = [(test_table.cell(i+1440,13).value - delta_x), (test_table.cell(i+1440,14).value - delta_y),
                   (test_table.cell(i+1440,13).value + delta_x), (test_table.cell(i+1440,14).value + delta_y)]
    region = img.crop(faceboundingbox)
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RM22 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RM22_net_test + '\\' + true_imagename)
    #for RM32:
    delta_x = math.ceil(0.12*(test_table.cell(i+1440,4).value - test_table.cell(i+1440,3).value))
    delta_y = math.ceil(0.12*(test_table.cell(i+1440,2).value - test_table.cell(i+1440,1).value))
    faceboundingbox = [(test_table.cell(i+1440,13).value - delta_x), (test_table.cell(i+1440,14).value - delta_y),
                   (test_table.cell(i+1440,13).value + delta_x), (test_table.cell(i+1440,14).value + delta_y)]
    region = img.crop(faceboundingbox)   #face bounding box image
    region_resize = region.resize((15,15), Image.ANTIALIAS)   #RM32 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_RM32_net_test + '\\' + true_imagename)
'''