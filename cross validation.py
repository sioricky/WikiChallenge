# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 11:18:46 2016

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


test_feature=np.load('test_feature.npy')[:,0:10]
test_target=np.load('test_target.npy')


'''linear regression'''
lr = linear_model.LinearRegression()
lr.fit(test_feature,test_target)
coeff=lr.coef_


'''random forest'''
rf = RandomForestRegressor(n_estimators=1000,max_depth=5)
#rf.fit(test_feature,test_target)


'''SVM'''
svm = svm.SVR(C=1)
#svm.fit(test_feature, test_target) 


'''Prediction'''
predicted = predicted = cross_val_predict(rf, test_feature, test_target, cv=10) #lr/rf/svm

'''Evaluation'''
mse = mean_squared_error(predicted,test_target)
rsme=math.sqrt(mse)
#n=len(test_target)
#d2=[]
#for i in range(0,n):
#    d2.append((math.log(1+predicted[i])-math.log(1+test_target[i]))**2)
#    
#e=math.sqrt(sum(d2)/n)

'''Plot'''
fig,ax = plt.subplots()
ax.scatter(test_target, predicted)
ax.plot([0,140000],[0,140000],'k--',lw=4) ## plot line y=x, the range can be changed
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