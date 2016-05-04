"""
@author: Yuan Liang
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import svm
#from sklearn import neighbors
#from sklearn import gaussian_process
#import cv2
import math
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor




data = pd.read_csv('case1_feature.tsv', sep='\t')
#feature = np.column_stack([data.ix[:,3].values, data.ix[:,4].values, data.ix[:,5].values, data.ix[:,6].values, data.ix[:,7].values, data.ix[:,8].values])
target=np.array(data.ix[:,2].values)
feature = np.column_stack([data.ix[:,5].values, data.ix[:,6].values, data.ix[:,7].values, data.ix[:,8].values])
  

model = linear_model.LinearRegression()
#model = linear_model.Ridge()
#model = linear_model.Lasso()
#model = linear_model.ElasticNet()
#model = linear_model.SGDRegressor()
#model = svm.SVR()
#model = svm.SVR(kernel='linear', C=1)
#    model = svm.SVR(kernel='rbf', C=1e3)
#    model = neighbors.NeighborsRegressor()
#    model = neighbors.NeighborsRegressor(n_neighbors=120, mode='mean')
#    model = gaussian_process.GaussianProcess()
#    model = cv2.GBTrees()




#model = RandomForestRegressor(n_estimators=15,max_depth=6)

#rf.fit(train_feature,train_target)
#predicted = rf.predict(test_feature)



# for cross validation
y_predict=cross_validation.cross_val_predict(model,feature,target,cv=10)
	# mean square error
mse=mean_squared_error(target,y_predict)
print 'RMS = %.6f\n' % mse

	# msele
n = len(y_predict)
sle = sum([math.pow(math.log(abs(y_predict[i]+1))-math.log(abs(target[i]+1)), 2) for i in range(n)])
rmsle = math.sqrt(sle/n)
print 'RMSLE = %.6f\n' % rmsle


#
#
## for model predicting
#	# for cv2 method
#model.train(feature, cv2.CV_ROW_SAMPLE, target, 
#                params={'weak_count':1000})  # 'subsample_portion':0.8, 'shrinkage':0.01
#    
#    
#    
#
#    # for scikits method
#model.fit(feature, target)
#prediction = model.predict(data)
#
#
#	# result
#coeff=model.coef_
#mse=mean_squared_error(data_target, prediction)
#print 'RMS = %.6f\n' % mse
#
#n=len(data_target)
#for i in range(n):
#    d2.append((math.log(1+prediction[i])-math.log(1+data_target[i]))**2)
#rmsle=math.sqrt(sum(d2)/n)
#print 'RMSLE = %.6f\n' % rmsle
