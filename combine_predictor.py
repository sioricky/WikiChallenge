# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 21:31:48 2016

@author: Administrator
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
import pandas as pd




from sklearn import neighbors
from sklearn import gaussian_process










data1 = pd.read_csv('./case2_features.tsv', sep='\t')
data2 = pd.read_csv('./case3_features.tsv', sep='\t')
data3 = pd.read_csv('./fuckfeature_part1.tsv', sep='\t')
data4 = pd.read_csv('./fuckfeature_part2.tsv', sep='\t')


predict_feature = np.column_stack([data1.ix[:,3].values, data1.ix[:,4].values, data1.ix[:,5].values, data3.ix[:,3].values, data3.ix[:,4].values, data3.ix[:,5].values])
predict_target = np.array(data1.ix[:,1].values)

train_feature = np.column_stack([data2.ix[:,3].values, data2.ix[:,4].values, data2.ix[:,5].values, data4.ix[:,3].values, data4.ix[:,4].values, data4.ix[:,5].values])
train_target = np.array(data2.ix[:,1].values)

#feature = np.column_stack([data.ix[:,3].values, data.ix[:,4].values, data.ix[:,5].values, data.ix[:,6].values, data.ix[:,7].values, data.ix[:,8].values])
#target=np.array(data.ix[:,2].values)
#feature = np.column_stack([data.ix[:,5].values, data.ix[:,6].values, data.ix[:,7].values, data.ix[:,8].values])
  

#model = linear_model.LinearRegression()
#model = linear_model.Ridge()
#model = linear_model.Lasso()
#model = linear_model.ElasticNet()
#model = linear_model.SGDRegressor()
#model = svm.SVR()
#model = svm.SVR(kernel='linear', C=1)
#model = svm.SVR(kernel='rbf', C=1e3)
#model = neighbors.RadiusNeighborsRegressor()
#model = neighbors.NeighborsRegressor(n_neighbors=120, mode='mean')
n_neighbors = 80
model = neighbors.KNeighborsClassifier(n_neighbors)
#model = gaussian_process.GaussianProcess()
#model = cv2.GBTrees()
#model = RandomForestRegressor(n_estimators=15,max_depth=6)






#rf.fit(train_feature,train_target)
#predicted = rf.predict(test_feature)



	# mean square error
#
#
# for model predicting
	# for cv2 method
#model.train(train_feature, cv2.CV_ROW_SAMPLE, train_target, 
 #               params={'weak_count':1000})  # 'subsample_portion':0.8, 'shrinkage':0.01
#    
#    
#    

#     for scikits method
model.fit(train_feature, train_target)
prediction = model.predict(predict_feature)


mse=mean_squared_error(predict_target,prediction)
rmse = math.sqrt(mse)
print 'RMS = %.6f\n' % rmse

	# msele
n = len(prediction)
sle = sum([math.pow(math.log(abs(prediction[i]+1))-math.log(abs(predict_target[i]+1)), 2) for i in range(n)])
rmsle = math.sqrt(sle/n)
print 'RMSLE = %.6f\n' % rmsle



#'''Plot'''
#fig,ax = plt.subplots()
#ax.scatter(predict_target, prediction)
#ax.plot([0,50],[0,50],'k--',lw=4) ## plot line y=x, the range can be changed
#ax.set_xlabel('Actual values')
#ax.set_ylabel('Fitted values')
#plt.show()
#
#residuals=abs(predict_target-prediction)
#fig2 = plt.subplot()
#plt.scatter(predict_target,residuals)
##plt.plot([0,1],[0,0],'k--',lw=4)  ## plot line y=x, the range can be changed
#plt.xlabel('Fitted values')
#plt.ylabel('Residuals')
#plt.show()