# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 17:21:15 2018

@author: zhaoyuzhi
"""

from PIL import Image
import xlrd
import numpy as np
import pandas as pd

#intend to get level 1 predicted landmarks for level 2 input
IMAGECASIA_test = "C:/Users/zhaoyuzhi/Desktop/CASIA_test"
CASIA_test_excel = xlrd.open_workbook('list.xlsx')
CASIA_test_table = CASIA_test_excel.sheet_by_index(0)
F1 = xlrd.open_workbook('F1.xlsx')
F1_table = F1.sheet_by_index(0)
EN1 = xlrd.open_workbook('EN1.xlsx')
EN1_table = EN1.sheet_by_index(0)
NM1 = xlrd.open_workbook('NM1.xlsx')
NM1_table = NM1.sheet_by_index(0)
F1_landmarks = np.zeros((4442,10), dtype=np.float32)
EN1_landmarks = np.zeros((4442,6), dtype=np.float32)
NM1_landmarks = np.zeros((4442,6), dtype=np.float32)
newlandmarks = np.zeros((4442,10), dtype=np.float32)
truelandmarks = np.zeros((4442,10), dtype=np.float32)

## get the location of newlandmarks - np.float32
for i in range(4442):
    F1_rawlandmarks = F1_table.row_slice(i+1, start_colx=1, end_colx=11)
    EN1_rawlandmarks = EN1_table.row_slice(i+1, start_colx=1, end_colx=7)
    NM1_rawlandmarks = NM1_table.row_slice(i+1, start_colx=1, end_colx=7)
    for j in range(10):
        F1_landmarks[i,j] = F1_rawlandmarks[j].value
    for k in range(6):
        EN1_landmarks[i,k] = EN1_rawlandmarks[k].value
        NM1_landmarks[i,k] = NM1_rawlandmarks[k].value

## build level 1 location - newlandmarks - LE, RE, N, LM, RM - normalized landmarks [0:39]
newlandmarks[:,0:4] = (F1_landmarks[:,0:4] + EN1_landmarks[:,0:4]) / 2
newlandmarks[:,4:6] = (F1_landmarks[:,4:6] + EN1_landmarks[:,4:6] + NM1_landmarks[:,0:2]) / 3
newlandmarks[:,6:10] = (F1_landmarks[:,6:10] + NM1_landmarks[:,2:6]) / 2

## build true landmarks according to the actual landmarks
for i in range(4442):
    imagename = CASIA_test_table.cell(i,0).value
    img = Image.open(IMAGECASIA_test + '/' + imagename,'r')
    [width, height] = img.size
    #get ten true newlandmarks(coordinates of LE,RE,N,LM,RM)
    for j in range(0,9,2):
        truelandmarks[i,j] = newlandmarks[i,j] * width / 39
    for k in range(1,10,2):
        truelandmarks[i,k] = newlandmarks[i,k] * height / 39

## save the predicted keypoints to excel
cache_truelandmarks_df = pd.DataFrame(truelandmarks)
cache_truelandmarks_df.columns = ['LEx','LEy','REx','REy','Nx','Ny','LMx','LMy','RMx','RMy']
writer = pd.ExcelWriter('level1.xlsx')
cache_truelandmarks_df.to_excel(writer,'sheet1')
writer.save()
