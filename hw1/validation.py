#! /usr/bin/python
#
# Author:	Cheng-Shih, Wong
# Stu. ID:	R04945028
# Email:	mob5566[at]gmail.com or r04945028[at]ntu.edu.tw
#

import numpy as np
import linreg_model as lrm
import decision_tree as dt
import csv
import time
import ensemble_model as em
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import RandomForestRegressor as RFR

# load training data
training_data = np.load('data/trainable.npy')

X = training_data[:, :-1]
y = training_data[:, -1]

# load testing data
Xtest = np.load('data/testable.npy')

# feature trimming

fmask = np.load('data/featureSelectMask.npy')

X = X[:, fmask]
Xtest = Xtest[:, fmask]

'''
X = X[:,90:]
Xtest = Xtest[:,90:]

tX = X[:, 6::18]
for i in [8, 9, 11]:
	tX = np.append(tX, X[:, i::18], axis=1)
X = tX

tXtest = Xtest[:, 6::18]
for i in [8, 9, 11]:
	tXtest = np.append(tXtest, Xtest[:, i::18], axis=1)
Xtest = tXtest

tX = X[:, 3::18]
for i in [5, 6, 8, 9, 11, 16, 17]:
	tX = np.append(tX, X[:, i::18], axis=1)
X = tX

tXtest = Xtest[:, 3::18]
for i in [5, 6, 8, 9, 11, 16, 17]:
	tXtest = np.append(tXtest, Xtest[:, i::18], axis=1)
Xtest = tXtest
'''

# setup learning models

#
# Linear Regression
#
models = [
[lrm.linreg(500, 0.1, True, 0.05, useAdagrad=True, useSGD=True, batchSize=30, useFeatureScaling=True, featureOrder=2)],
[lrm.linreg(500, 1, True, 0.05, useAdagrad=True, useSGD=True, batchSize=30, useFeatureScaling=True, featureOrder=2)],
[lrm.linreg(500, 10, True, 0.05, useAdagrad=True, useSGD=True, batchSize=30, useFeatureScaling=True, featureOrder=2)]]

#
# Decision Tree
#
# models = [
# [dt.dtree(5), dt.dtree(5, 0.3), dt.dtree(5, 0.5)],
# [dt.dtree(10), dt.dtree(10, 0.3), dt.dtree(10, 0.5)], 
# [dt.dtree(), dt.dtree(None, 0.3), dt.dtree(None, 0.5)],
# [dt.dtree(), dt.dtree(None, 0.3), dt.dtree(None, 0.5)]]


#
# Random Forest
#
# models = [
# [em.random_forest(dt.dtree, (None, 0.33), 10, 1)],
# [em.random_forest(dt.dtree, (None, 0.33), 15, 1)],
# [em.random_forest(dt.dtree, (None, 0.33), 30, 1)]]

#
# Random Forest from sklearn
#
# models = [
# [DTR(max_depth=None, max_features=None, random_state=0), RFR(10, max_features=None, max_depth=None, random_state=0)],
# [DTR(max_depth=None, max_features=None, random_state=0), RFR(30, max_features=None, max_depth=None, random_state=0)],
# [DTR(max_depth=None, max_features=None, random_state=0), RFR(50, max_features=None, max_depth=None, random_state=0)],
# [DTR(max_depth=None, max_features=None, random_state=0), RFR(100, max_features=None, max_depth=None, random_state=0)]]

valid = lrm.validation(models, X, y, lrm.RMSE, 0.3 )

print('Validation...')
tstart = time.time()
valid.scores()
print('Done!')
print('Training cost %.3f seconds!' % (time.time()-tstart))

# output RMSE of insample
print('RMSE = %.3f' % lrm.RMSE(valid.getBestModel(), X, y))

# plot the learning curve
import matplotlib.pyplot as plt
plt.plot(np.arange(len(models[0][0].hist_e)), models[0][0].hist_e, 'x-', label='eta = 0.1')
plt.plot(np.arange(len(models[1][0].hist_e)), models[1][0].hist_e, 'o-', label='eta = 1')
plt.plot(np.arange(len(models[2][0].hist_e)), models[2][0].hist_e, '*-', label='eta = 10')
plt.legend(loc='upper right')
plt.show()

# make prediction
yout = valid.getBestModel().predict(Xtest)

outputfile = open('linear_regression.csv', 'wb')
csv_output = csv.writer(outputfile)

csv_output.writerow(['id', 'value'])

for idx, value in enumerate(yout):
	csv_output.writerow(['id_'+str(idx), value])

outputfile.close()

