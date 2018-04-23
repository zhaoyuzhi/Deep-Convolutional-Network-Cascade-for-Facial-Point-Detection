# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 17:52:19 2018

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
LE31 = xlrd.open_workbook('LE31.xlsx')
LE31_table = LE31.sheet_by_index(0)
LE32 = xlrd.open_workbook('LE32.xlsx')
LE32_table = LE32.sheet_by_index(0)
RE31 = xlrd.open_workbook('RE31.xlsx')
RE31_table = RE31.sheet_by_index(0)
RE32 = xlrd.open_workbook('RE32.xlsx')
RE32_table = RE32.sheet_by_index(0)
N31 = xlrd.open_workbook('N31.xlsx')
N31_table = N31.sheet_by_index(0)
N32 = xlrd.open_workbook('N32.xlsx')
N32_table = N32.sheet_by_index(0)
LM31 = xlrd.open_workbook('LM31.xlsx')
LM31_table = LM31.sheet_by_index(0)
LM32 = xlrd.open_workbook('LM32.xlsx')
LM32_table = LM32.sheet_by_index(0)
RM31 = xlrd.open_workbook('RM31.xlsx')
RM31_table = RM31.sheet_by_index(0)
RM32 = xlrd.open_workbook('RM32.xlsx')
RM32_table = RM32.sheet_by_index(0)
LE_landmarks = np.zeros((4442,2), dtype=np.float32)
RE_landmarks = np.zeros((4442,2), dtype=np.float32)
N_landmarks = np.zeros((4442,2), dtype=np.float32)
LM_landmarks = np.zeros((4442,2), dtype=np.float32)
RM_landmarks = np.zeros((4442,2), dtype=np.float32)
newlandmarks = np.zeros((4442,10), dtype=np.float32)
truelandmarks = np.zeros((4442,10), dtype=np.float32)

## get the location of newlandmarks - np.float32
for i in range(4442):
    LE31_rawlandmarks = LE31_table.row_slice(i+1, start_colx=1, end_colx=3)
    LE32_rawlandmarks = LE32_table.row_slice(i+1, start_colx=1, end_colx=3)
    RE31_rawlandmarks = RE31_table.row_slice(i+1, start_colx=1, end_colx=3)
    RE32_rawlandmarks = RE32_table.row_slice(i+1, start_colx=1, end_colx=3)
    N31_rawlandmarks = N31_table.row_slice(i+1, start_colx=1, end_colx=3)
    N32_rawlandmarks = N32_table.row_slice(i+1, start_colx=1, end_colx=3)
    LM31_rawlandmarks = LM31_table.row_slice(i+1, start_colx=1, end_colx=3)
    LM32_rawlandmarks = LM32_table.row_slice(i+1, start_colx=1, end_colx=3)
    RM31_rawlandmarks = RM31_table.row_slice(i+1, start_colx=1, end_colx=3)
    RM32_rawlandmarks = RM32_table.row_slice(i+1, start_colx=1, end_colx=3)
    for j in range(2):
        LE31 = LE31_rawlandmarks[j].value
        LE32 = LE32_rawlandmarks[j].value
        LE_landmarks[i,j] = (LE31 + LE32) / 2
        RE31 = RE31_rawlandmarks[j].value
        RE32 = RE32_rawlandmarks[j].value
        RE_landmarks[i,j] = (RE31 + RE32) / 2
        N31 = N31_rawlandmarks[j].value
        N32 = N32_rawlandmarks[j].value
        N_landmarks[i,j] = (N31 + N32) / 2
        LM31 = LM31_rawlandmarks[j].value
        LM32 = LM32_rawlandmarks[j].value
        LM_landmarks[i,j] = (LM31 + LM32) / 2
        RM31 = RM31_rawlandmarks[j].value
        RM32 = RM32_rawlandmarks[j].value
        RM_landmarks[i,j] = (RM31 + RM32) / 2

## build level 2 location - newlandmarks - LE, RE, N, LM, RM - normalized landmarks [0:15]
newlandmarks[:,0:2] = LE_landmarks
newlandmarks[:,2:4] = RE_landmarks
newlandmarks[:,4:6] = N_landmarks
newlandmarks[:,6:8] = LM_landmarks
newlandmarks[:,8:10] = RM_landmarks

## build true landmarks according to the actual landmarks
for i in range(4442):
    imagename = CASIA_test_table.cell(i,0).value
    img = Image.open(IMAGECASIA_test + '/' + imagename,'r')
    [width, height] = img.size
    #get ten true newlandmarks(coordinates of LE,RE,N,LM,RM)
    for j in range(0,9,2):
        truelandmarks[i,j] = newlandmarks[i,j] * width / 15
    for k in range(1,10,2):
        truelandmarks[i,k] = newlandmarks[i,k] * height / 15

## save the predicted keypoints to excel
cache_truelandmarks_df = pd.DataFrame(truelandmarks)
cache_truelandmarks_df.columns = ['LEx','LEy','REx','REy','Nx','Ny','LMx','LMy','RMx','RMy']
writer = pd.ExcelWriter('level3.xlsx')
cache_truelandmarks_df.to_excel(writer,'sheet1')
writer.save()
