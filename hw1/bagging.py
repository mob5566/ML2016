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

# feature trimming
'''
tX = X[:, 6::18]
for i in [8, 9, 11]:
	tX = np.append(tX, X[:, i::18], axis=1)
X = tX

tXtest = Xtest[:, 6::18]
for i in [8, 9, 11]:
	tXtest = np.append(tXtest, Xtest[:, i::18], axis=1)
Xtest = tXtest
'''

tX = X[:, 3::18]
for i in [5, 6, 7, 8, 9, 11, 12, 16, 17]:
	tX = np.append(tX, X[:, i::18], axis=1)
X = tX

tXtest = Xtest[:, 3::18]
for i in [5, 6, 7, 8, 9, 11, 12, 16, 17]:
	tXtest = np.append(tXtest, Xtest[:, i::18], axis=1)
Xtest = tXtest

# bagging

paras = (500, 1, True, 0.1, True, 30, 3, True, True)
bag = em.bagging(lrm.linreg, paras, 100)

print('Training...')
tstart = time.time()
bag.fit(X, y)
print('Done!')
print('Training cost %.3f seconds!' % (time.time()-tstart))

# output RMSE of insample
print('RMSE = %.3f' % lrm.RMSE(bag, X, y))

# make prediction
yout = bag.predict(Xtest)

outputfile = open('bagging.csv', 'wb')
csv_output = csv.writer(outputfile)

csv_output.writerow(['id', 'value'])

for idx, value in enumerate(yout):
	csv_output.writerow(['id_'+str(idx), value])

outputfile.close()

