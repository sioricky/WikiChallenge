# -*- coding: utf-8 -*-
"""
Created on Sat Mar 05 18:33:40 2016

@author: Administrator
"""


import pandas as pd

data1 = []
data1 = pd.read_csv('datafeature.tsv', sep='\t')

data2 = []
data2 = pd.read_csv('case1_featurepart2_extend.tsv', sep='\t')


data = pd.DataFrame({'user_id': data1['user_id'].values,
 'train_target': data1['train_target'].values,
 'article_number': data1['articles_edited_numbers'].values,
 'edit_duration': data2['user_durating'].values,
 'recent_edit_times': data1['recent_edit_times'].values,
 'weighted_edit_date': data2['user_recent_editdata'].values,
 'user_average_edit_data': data2['user_averagedate'].values,
 'strd_edit_derviation': data1['strd_edit_derivation'].values,
})

frame = ['user_id', 'train_target', 'article_number', 'edit_duration', 'recent_edit_times', 'weighted_edit_date', 'user_average_edit_data', 'strd_edit_derviation']
data = data[frame]
data.to_csv('case1_feature.tsv', sep='\t')



import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error


# train and vidation with 10-fold test
train_feature = np.column_stack([data.ix[:,2].values, data.ix[:,3].values, data.ix[:,4].values, data.ix[:,5].values, data.ix[:,6].values, data.ix[:,7].values])
# train_feature=np.asarray(train_feature).T
train_target=np.array(data.ix[:,1].values)

regressor = linear_model.LinearRegression()
y_predict=cross_validation.cross_val_predict(regressor,train_feature,train_target,cv=10)
mse=mean_squared_error(train_target,y_predict)
print(mse)

