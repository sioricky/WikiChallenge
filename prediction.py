# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 07:39:29 2016

@author: ningwang
"""
import math
import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

t_feature=np.load('train_feature.npy')
t_target=np.load('train_target.npy')
test_feature=np.load('test_feature.npy')[:,0:6]
test_target=np.load('test_target.npy')

train_feature=[]
train_target=[]
t_feature=list(t_feature)
t_target=list(t_target)

for i in range(len(t_feature)):
    if t_feature[i][6]>=1:
        train_feature.append(t_feature[i])
        train_target.append(t_target[i])

train_feature=np.array(train_feature)[:,0:6]
train_target=np.array(train_target)

'''linear regression'''
#lr = linear_model.LinearRegression()
#lr.fit(train_feature,train_target)
#coeff=lr.coef_
#predicted = lr.predict(test_feature)


'''random forest'''
rf = RandomForestRegressor(n_estimators=12,max_depth=8)
rf.fit(train_feature,train_target)
predicted = rf.predict(test_feature)


'''SVM'''
#clf = svm.SVR(C=10)
#clf.fit(train_feature, train_target) 
#predicted = clf.predict(test_feature)

'''Evaluation'''
mse = mean_squared_error(predicted,test_target)
rsme=math.sqrt(mse)
n=len(test_target)
d2=[]
for i in range(0,n):
    d2.append((math.log(1+predicted[i])-math.log(1+test_target[i]))**2)
    
e=math.sqrt(sum(d2)/n)

'''Plot'''
fig,ax = plt.subplots()
ax.scatter(test_target, predicted)
ax.plot([0,50],[0,50],'k--',lw=4) ## plot line y=x, the range can be changed
ax.set_xlabel('Actual values')
ax.set_ylabel('Fitted values')
plt.show()

residuals=abs(test_target-predicted)
fig2 = plt.subplot()
plt.scatter(predicted,residuals)
#plt.plot([0,1],[0,0],'k--',lw=4)  ## plot line y=x, the range can be changed
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.show()

