#! /usr/bin/python
#
# Author:	Cheng-Shih, Wong
# Stu. ID:	R04945028
# Email:	mob5566[at]gmail.com or r04945028[at]ntu.edu.tw
#

import numpy as np
import linreg_model as lrm
import ensemble_model as em
import decision_tree as dt
import csv
import time

# load training data
training_data = np.load('data/trainable.npy')

X = training_data[:, :-1]
y = training_data[:, -1]

# load testing data
Xtest = np.load('data/testable.npy')

rf = em.random_forest(10, max_depth=None, max_features=None, sampleNum=None)

print('Training...')
tstart = time.time()
rf.fit(X, y)
print('Done!')
print('Training cost %.3f seconds!' % (time.time()-tstart))

# output RMSE of insample
print('RMSE = %.3f' % lrm.RMSE(rf, X, y))

# make prediction
yout = rf.predict(Xtest)

outputfile = open('random_forest.csv', 'wb')
csv_output = csv.writer(outputfile)

csv_output.writerow(['id', 'value'])

for idx, value in enumerate(yout):
	csv_output.writerow(['id_'+str(idx), value])

outputfile.close()

