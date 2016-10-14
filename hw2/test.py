#! /usr/bin/python
#
# Author:	Cheng-Shih, Wong
# Stu. ID:	R04945028
# Email:	mob5566[at]gmail.com or r04945028[at]ntu.edu.tw
#

import numpy as np
import logreg_model as lgr
import csv
import time

# load training data
training_data = np.load('spam_data/training_data.npy')

X = training_data[:, :-1]
y = training_data[:, -1]

# load testing data
Xtest = np.load('spam_data/testing_data.npy')

# setup linear regression model
model = lgr.logreg(3000, 0.1, True, 0.1, useAdagrad=True, useSGD=True, batchSize=30)

print('Training...')
tstart = time.time()
model.fit(X, y)
print('Done!')
print('Training cost %.3f seconds!' % (time.time()-tstart))

print 'Ein\n', lgr.mismatch(model, X, y)

# make prediction
yout = (model.predict(Xtest)>0.5).astype(np.int)

outputfile = open('prediction.csv', 'wb')
csv_output = csv.writer(outputfile)

csv_output.writerow(['id', 'label'])

for idx, value in enumerate(yout, 1):
	csv_output.writerow([idx, value])

outputfile.close()

