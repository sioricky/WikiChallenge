# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 07:39:29 2016

@author: Gaoxiang
"""
import math
import numpy as np

train_feature=np.load('train_feature.npy')
train_target=np.load('train_target.npy')
test_feature=np.load('test_feature.npy')
test_target=np.load('test_target.npy')


print(train_feature.shape[0])
print(train_feature.shape[1])
col=train_feature.shape[1]
row=train_feature.shape[0]
train_f_log=np.zeros([row,col])
for i in range(0,row):
    for j in range(0,col):
        train_f_log[i][j]=math.log1p(train_feature[i][j])


col=test_feature.shape[1]
row=test_feature.shape[0]
test_f_log=np.zeros([row,col])
for i in range(0,row):
    for j in range(0,col):
        test_f_log[i][j]=math.log1p(test_feature[i][j])

n=len(train_target)
train_t_log=np.zeros(n)
for i in range(0,n):
    train_t_log[i]=math.log1p(train_target[i])

n=len(test_target)
test_t_log=np.zeros(n)
for i in range(0,n):
    test_t_log[i]=math.log1p(test_target[i])
np.save("train_feature_log",train_f_log)
np.save("test_feature_log",test_f_log)
np.save("train_target_log",train_t_log)
np.save("test_target_log",test_t_log)