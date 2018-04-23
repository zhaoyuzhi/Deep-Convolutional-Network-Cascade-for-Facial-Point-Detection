# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 16:25:26 2018

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
LE21 = xlrd.open_workbook('LE21.xlsx')
LE21_table = LE21.sheet_by_index(0)
LE22 = xlrd.open_workbook('LE22.xlsx')
LE22_table = LE22.sheet_by_index(0)
RE21 = xlrd.open_workbook('RE21.xlsx')
RE21_table = RE21.sheet_by_index(0)
RE22 = xlrd.open_workbook('RE22.xlsx')
RE22_table = RE22.sheet_by_index(0)
N21 = xlrd.open_workbook('N21.xlsx')
N21_table = N21.sheet_by_index(0)
N22 = xlrd.open_workbook('N22.xlsx')
N22_table = N22.sheet_by_index(0)
LM21 = xlrd.open_workbook('LM21.xlsx')
LM21_table = LM21.sheet_by_index(0)
LM22 = xlrd.open_workbook('LM22.xlsx')
LM22_table = LM22.sheet_by_index(0)
RM21 = xlrd.open_workbook('RM21.xlsx')
RM21_table = RM21.sheet_by_index(0)
RM22 = xlrd.open_workbook('RM22.xlsx')
RM22_table = RM22.sheet_by_index(0)
LE_landmarks = np.zeros((4442,2), dtype=np.float32)
RE_landmarks = np.zeros((4442,2), dtype=np.float32)
N_landmarks = np.zeros((4442,2), dtype=np.float32)
LM_landmarks = np.zeros((4442,2), dtype=np.float32)
RM_landmarks = np.zeros((4442,2), dtype=np.float32)
newlandmarks = np.zeros((4442,10), dtype=np.float32)
truelandmarks = np.zeros((4442,10), dtype=np.float32)

## get the location of newlandmarks - np.float32
for i in range(4442):
    LE21_rawlandmarks = LE21_table.row_slice(i+1, start_colx=1, end_colx=3)
    LE22_rawlandmarks = LE22_table.row_slice(i+1, start_colx=1, end_colx=3)
    RE21_rawlandmarks = RE21_table.row_slice(i+1, start_colx=1, end_colx=3)
    RE22_rawlandmarks = RE22_table.row_slice(i+1, start_colx=1, end_colx=3)
    N21_rawlandmarks = N21_table.row_slice(i+1, start_colx=1, end_colx=3)
    N22_rawlandmarks = N22_table.row_slice(i+1, start_colx=1, end_colx=3)
    LM21_rawlandmarks = LM21_table.row_slice(i+1, start_colx=1, end_colx=3)
    LM22_rawlandmarks = LM22_table.row_slice(i+1, start_colx=1, end_colx=3)
    RM21_rawlandmarks = RM21_table.row_slice(i+1, start_colx=1, end_colx=3)
    RM22_rawlandmarks = RM22_table.row_slice(i+1, start_colx=1, end_colx=3)
    for j in range(2):
        LE21 = LE21_rawlandmarks[j].value
        LE22 = LE22_rawlandmarks[j].value
        LE_landmarks[i,j] = (LE21 + LE22) / 2
        RE21 = RE21_rawlandmarks[j].value
        RE22 = RE22_rawlandmarks[j].value
        RE_landmarks[i,j] = (RE21 + RE22) / 2
        N21 = N21_rawlandmarks[j].value
        N22 = N22_rawlandmarks[j].value
        N_landmarks[i,j] = (N21 + N22) / 2
        LM21 = LM21_rawlandmarks[j].value
        LM22 = LM22_rawlandmarks[j].value
        LM_landmarks[i,j] = (LM21 + LM22) / 2
        RM21 = RM21_rawlandmarks[j].value
        RM22 = RM22_rawlandmarks[j].value
        RM_landmarks[i,j] = (RM21 + RM22) / 2

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
writer = pd.ExcelWriter('level2.xlsx')
cache_truelandmarks_df.to_excel(writer,'sheet1')
writer.save()
