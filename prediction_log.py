# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 07:39:29 2016

@author: Gaoxiang
"""
import math
import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors

train_feature=np.load('train_feature_log.npy')
train_target=np.load('train_target_log.npy')
test_feature=np.load('test_feature_log.npy')
test_target=np.load('test_target_log.npy')


'''GradientBoosting'''
'''
rf = GradientBoostingRegressor(n_estimators=135, learning_rate=0.01,max_depth=1, random_state=0, loss='ls')
rf.fit(train_feature,train_target)
predicted = rf.predict(test_feature)
'''
'''linear regression'''

#poly = PolynomialFeatures(1)
#train_feature=poly.fit_transform(train_feature)
#test_feature=poly.fit_transform(test_feature)

lr = linear_model.LinearRegression()
lr.fit(train_feature,train_target)
coeff=lr.coef_
predicted = lr.predict(test_feature)


'''randomforest'''

'''
rf = RandomForestRegressor(n_estimators=55,max_depth=8)
rf.fit(train_feature,train_target)
predicted = rf.predict(test_feature)
'''

'''
rf = linear_model.Lasso()
rf.fit(train_feature,train_target)
predicted = rf.predict(test_feature)
#model = linear_model.Ridge()
#model = linear_model.Lasso()
'''
'''
rf = neighbors.KNeighborsClassifier(80)
rf.fit(train_feature,train_target)
predicted = rf.predict(test_feature)
'''
'''SVM'''
'''
clf = svm.SVR(kernel='linear')
clf.fit(train_feature, train_target) 
predicted = clf.predict(test_feature)
'''

'''evaluation'''
msle = mean_squared_error(predicted,test_target)
rmsle = math.sqrt(msle)

target_a=np.load('test_target.npy')
n=len(target_a)
d=[]
prediction_a=np.zeros(n)
for i in range(0,n):
    prediction_a[i]=math.exp(predicted[i])-1
mse=mean_squared_error(prediction_a,target_a)
rmse=math.sqrt(mse)

fig,ax = plt.subplots()
ax.scatter(target_a, prediction_a)
ax.plot([0,400],[0,400],'k--',lw=4) ## plot line y=x, the range can be changed
ax.set_xlabel('Actual values')
ax.set_ylabel('Fitted values')
plt.show()

residuals=abs(target_a-prediction_a)
fig2 = plt.subplot()
plt.scatter(prediction_a,residuals)
#plt.plot([0,1],[0,0],'k--',lw=4)  ## plot line y=x, the range can be changed
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.show()

