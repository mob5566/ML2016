#! /usr/bin/python
#
# Author:	Cheng-Shih, Wong
# Stu. ID:	R04945028
# Email:	mob5566[at]gmail.com or r04945028[at]ntu.edu.tw
#

import numpy as np
import logreg_model as lgr
import ensemble_model as em
import csv
import time
import sys
import cPickle as pickle

if len(sys.argv)!=3 or\
	sys.argv[1][-4:]!='.csv':

	print 'In train.py:'
	print '\tUsage: python train.sh training_data.csv output_model'
	exit()

fname = sys.argv[1][:-4]

# load training data
training_data = np.load(fname+'.npy')

X = training_data[:, :-1]
y = training_data[:, -1]

# setup linear regression model
model = em.random_forest(100, max_depth=20, max_features=25)

print('Training...')
tstart = time.time()
model.fit(X, y)
print('Done!')
print('Training cost %.3f seconds!' % (time.time()-tstart))

print 'Ein\n', lgr.mismatch(model, X, y)

# save the trained model to pickle file
with open(sys.argv[2]+'.pkl', 'wb') as output:
	pickle.dump(model, output, -1)
