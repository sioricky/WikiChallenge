# -*- coding: utf-8 -*-
"""
Created on Sun Mar 06 02:18:19 2016

@author: Yuan Liang
"""

import pandas as pd
import numpy as np

data = pd.read_csv('case1_feature.tsv', sep='\t')
feature = np.column_stack([data.ix[:,3].values, data.ix[:,4].values, data.ix[:,5].values, data.ix[:,6].values, data.ix[:,7].values, data.ix[:,8].values])
target=np.array(data.ix[:,2].values)


'''Recursive Feature Elimination'''
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
import numpy as np

train_feature = np.column_stack([data.ix[:,2].values, data.ix[:,3].values, data.ix[:,4].values, data.ix[:,5].values, data.ix[:,6].values, data.ix[:,7].values])
train_target=np.array(data.ix[:,1].values)

estimator = SVR(kernel="linear")
selector = RFECV(estimator, step=1, cv=8)
selector = selector.fit(train_feature, train_target)

selector.support_ 
selector.ranking_
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
plt.show()



'''Univariate Feature Selection'''
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.feature_selection import SelectPercentile, f_classif

train_feature = np.column_stack([data.ix[:,2].values, data.ix[:,3].values, data.ix[:,4].values, data.ix[:,5].values, data.ix[:,6].values, data.ix[:,7].values])
train_target=np.array(data.ix[:,1].values)

selector = SelectPercentile(f_classif, percentile=10)
selector.fit(train_feature, train_target)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()


clf = svm.SVC(kernel='linear')
clf.fit(train_feature, train_target)

svm_weights = (clf.coef_ ** 2).sum(axis=0)
svm_weights /= svm_weights.max()


clf_selected = svm.SVC(kernel='linear')
clf_selected.fit(selector.transform(train_feature), train_target)

svm_weights_selected = (clf_selected.coef_ ** 2).sum(axis=0)
svm_weights_selected /= svm_weights_selected.max()



''' tree method'''
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier


forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(train_feature, train_target)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(train_feature.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(train_feature.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(train_feature.shape[1]), indices)
plt.xlim([-1, train_feature.shape[1]])
plt.show()



