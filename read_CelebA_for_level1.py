# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 09:37:38 2018

@author: zhaoyuzhi
"""

#This py file is to process CelebA images to train the networks
#CelebA image lib has 202599 images with 5 key points label
#I think CelebA could be regarded as training images and LFPW/BioID could be regarded as testing images
from PIL import Image
import math
import xlrd

#It is important that before you use this code, you should modify the image URLs
IMAGEURL = "G:\\CelebA\\Img\\img_align_celeba"

IMAGESAVEURL_F1_CelebA = "G:\\CelebA\\Img\\img_align_celeba_F1"
IMAGESAVEURL_EN1_CelebA = "G:\\CelebA\\Img\\img_align_celeba_EN1"
IMAGESAVEURL_NM1_CelebA = "G:\\CelebA\\Img\\img_align_celeba_NM1"

bbox_excel = xlrd.open_workbook('CelebA_faceboundingbox.xlsx')
bbox_table = bbox_excel.sheet_by_index(0)
#landmarks_excel = xlrd.open_workbook('CelebA_align_landmarks.xlsx')
#landmarks_table = landmarks_excel.sheet_by_index(0)

#this is for CelebA
for i in range(204599):
    imagename = bbox_table.cell(i+1,0).value
    img = Image.open(IMAGEURL + '\\' + imagename,'r')
    #input images (width = 178, height = 218)
    #min(LEx) = 56, min(LEy) = 98, max(REx) = 124, min(REy) = 95, min(LMx) = 57, max(LMy) = 174, max(REx) = 120, max(REy) = 173
    #the range of Nx is [57,121], the range of Ny is [93,156]
    #Conclusion: min(left) = 56, min(top) = 95, max(right) = 121, max(bottom) = 174
    #I choose the coordinate of left_top corner as (44,90) and right_bottom corner as (133,179)
    #at first crop a square region for test, length is 89
    #that is faceboundingbox = [44,90,133,179]
    #output/crop face_bounding_box(left,top,right,bottom)
    #img.resize((width, height))
    #for F1:
    faceboundingbox_F1 = [44,90,133,179]
    region = img.crop(faceboundingbox_F1)   #face bounding box image
    region_resize = region.resize((39,39), Image.ANTIALIAS)   #F1 image
    region_resize_grey = region_resize.convert('L')   #convert to grey image
    region_resize_grey.save(IMAGESAVEURL_F1_CelebA + '\\' + imagename)
    #for EN1:
    faceboundingbox_EN1 = [50,90,127,150]
    region_EN1 = img.crop(faceboundingbox_EN1)
    region_EN1_resize = region_EN1.resize((39,31), Image.ANTIALIAS)   #EN1 image
    region_EN1_resize_grey = region_EN1_resize.convert('L')   #convert to grey image
    region_EN1_resize_grey.save(IMAGESAVEURL_EN1_CelebA + '\\' + imagename)
    #for NM1:
    faceboundingbox_NM1 = [50,119,127,179]
    region_NM1 = img.crop(faceboundingbox_NM1)
    region_NM1_resize = region_NM1.resize((39,31), Image.ANTIALIAS)   #EN1 image
    region_NM1_resize_grey = region_NM1_resize.convert('L')   #convert to grey image
    region_NM1_resize_grey.save(IMAGESAVEURL_NM1_CelebA + '\\' + imagename)
