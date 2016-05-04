# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 12:15:25 2016

@author: ningwang
"""

import numpy as np
import csv

'''train'''
with open('training.tsv') as f:
    reader = csv.reader(f,delimiter="\t")
    data_list = [line for line in reader]
    data = np.array(data_list)

data = data[1:,:]
np.save('train_all',data)

train = data[:,0:5]
np.save('train',train)

#'''test'''
#with open('validation.tsv') as f:
#    reader = csv.reader(f,delimiter="\t")
#    data_list = [line for line in reader]
#    data = np.array(data_list)
#
#data = data[1:,:]
#np.save('test_all',data)
#
#train = data[:,0:5]
#np.save('test',train)
#
#'''solution'''
#with open('validation_solutions.csv','r') as f:
#    reader = csv.reader(f)
#    data_list = [line for line in reader]
#    data = np.array(data_list)
#
#solution= data[1:,:]
#np.save('solution',solution)