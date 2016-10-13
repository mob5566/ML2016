#! /usr/bin/python
#
# Author:	Cheng-Shih, Wong
# Stu. ID:	R04945028
# Email:	mob5566[at]gmail.com or r04945028[at]ntu.edu.tw
#

import numpy as np
import linreg_model as lrm
import csv
import time

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

# setup learning models
models = [
[lrm.linreg(500, 1, True, 0.1, useAdagrad=True, useSGD=True, batchSize=30, useFeatureScaling=True, featureOrder=1)],
[lrm.linreg(500, 1, True, 0.05, useAdagrad=True, useSGD=True, batchSize=30, useFeatureScaling=True, featureOrder=2)],
[lrm.linreg(500, 1, True, 0.1, useAdagrad=True, useSGD=True, batchSize=30, useFeatureScaling=True, featureOrder=3)]]

cv = lrm.cross_valid(models, X, y, lrm.RMSE, 5)

print('Cross validation...')
tstart = time.time()
cv.scores()
print('Done!')
print('Training cost %.3f seconds!' % (time.time()-tstart))

# output RMSE of insample
print('RMSE = %.3f' % lrm.RMSE(cv.getBestModel(), X, y))

# make prediction
yout = cv.getBestModel().predict(Xtest)

outputfile = open('cv_linear_regression.csv', 'wb')
csv_output = csv.writer(outputfile)

csv_output.writerow(['id', 'value'])

for idx, value in enumerate(yout):
	csv_output.writerow(['id_'+str(idx), value])

outputfile.close()

