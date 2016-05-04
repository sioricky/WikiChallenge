# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 17:04:37 2016

@author: ningwang
"""
import math
import numpy as np
from datetime import datetime


train = np.load('train.npy')

for line in train:
    t = line[4]
    time = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
    dt = (2010-time.year)*12 + 9-time.month + (1-time.day)/30.0
    line[4] = dt

train = train.astype(np.float)
train_user = np.unique(train[:,0])


train_att=[]
for i in range(0,10):
    train_att.append(1)

train_att[0] = list(train[(train[:,4]>5)[:] * (train[:,4]<=5.3)[:]] [:,0])  #0.3
train_att[1] = list(train[(train[:,4]>5)[:] * (train[:,4]<=5.6)[:]] [:,0])  #0.6    
train_att[2] = list(train[(train[:,4]>5)[:] * (train[:,4]<=6)[:]] [:,0])  #1
train_att[3] = list(train[(train[:,4]>5)[:] * (train[:,4]<=7)[:]] [:,0])  #2
train_att[4] = list(train[(train[:,4]>5)[:] * (train[:,4]<=9)[:]] [:,0])  #4
train_att[5] = list(train[(train[:,4]>5)[:] * (train[:,4]<=13)[:]] [:,0])  #8
train_att[6] = list(train[(train[:,4]>5)[:] * (train[:,4]<=21)[:]] [:,0])  #16
train_att[7] = list(train[(train[:,4]>5)[:] * (train[:,4]<=37)[:]] [:,0])  #32
train_att[8] = list(train[(train[:,4]>5)[:] * (train[:,4]<=69)[:]] [:,0])  #64
train_att[9] = list(train[(train[:,4]>5)[:] * (train[:,4]<=113)[:]] [:,0]) #108

train_tar = list(train[(train[:,4]<=5)[:] * (train[:,4]>=0)[:]] [:,0])

train_feature=[]
train_target=[] 
 
for i in train_user:
    for j in range(0,10):
        train_feature.append(train_att[j].count(i))
    train_target.append(train_tar.count(i))
 
train_feature=np.array(train_feature)
train_target=np.array(train_target)
train_feature=train_feature.reshape((len(train_user),10))

#np.save('train_user',train_user)
#np.save('train_feature',train_feature)
#np.save('train_target',train_target)






