# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 14:52:06 2016

@author: ningwang
"""

import math
import numpy as np
from datetime import datetime


test=np.load('test.npy')
solution=np.load('solution.npy')

for line in test:
    t = line[4]
    time = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
    dt = (2008-time.year)*12 + 1-time.month + (1-time.day)/30.0
    line[4] = dt

test = test.astype(np.float)
solution = solution.astype(np.float)
test_user = np.unique(test[:,0])

test_att=[]
for i in range(0,10):
    test_att.append(1)


test_att[0] = list(test[(test[:,4]>0)[:] * (test[:,4]<=0.3)[:]] [:,0])  #0.3
test_att[1] = list(test[(test[:,4]>0)[:] * (test[:,4]<=0.6)[:]] [:,0])  #0.6
test_att[2] = list(test[(test[:,4]>0)[:] * (test[:,4]<=1)[:]] [:,0])  #1
test_att[3] = list(test[(test[:,4]>0)[:] * (test[:,4]<=2)[:]] [:,0])  #2
test_att[4] = list(test[(test[:,4]>0)[:] * (test[:,4]<=4)[:]] [:,0])  #4
test_att[5] = list(test[(test[:,4]>0)[:] * (test[:,4]<=8)[:]] [:,0])  #8
test_att[6] = list(test[(test[:,4]>0)[:] * (test[:,4]<=16)[:]] [:,0])  #16
test_att[7] = list(test[(test[:,4]>0)[:] * (test[:,4]<=32)[:]] [:,0])  #32
test_att[8] = list(test[(test[:,4]>0)[:] * (test[:,4]<=64)[:]] [:,0])  #64
test_att[9] = list(test[(test[:,4]>0)[:] * (test[:,4]<=108)[:]] [:,0]) #108


test_feature=[]
test_target=[] 
for i in test_user:
    for j in range(0,10):
        test_feature.append(test_att[j].count(i))
    n=int(solution[solution[:,0]==i][:,1])
    test_target.append(n)


test_feature=np.array(test_feature)
test_feature=test_feature.reshape((len(test_user),10))    
test_target=np.array(test_target)

#np.save('test_user',test_user)
#np.save('test_feature',test_feature)
#np.save('test_target',test_target)















