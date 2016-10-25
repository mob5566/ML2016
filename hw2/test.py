#! /usr/bin/python
#
# Author:	Cheng-Shih, Wong
# Stu. ID:	R04945028
# Email:	mob5566[at]gmail.com or r04945028[at]ntu.edu.tw
#

import csv
import numpy as np
import time
import sys
import cPickle as pickle

if len(sys.argv)!=4 or\
	sys.argv[2][-4:]!='.csv':
	
	print 'In test.py:'
	print '\tUsage: test.py model test_data.csv predict_output'
	exit()

model_name = sys.argv[1]
test_fname = sys.argv[2][:-4]
predict_output = sys.argv[3]

# load model
with open(model_name, 'rb') as model_input:
	model = pickle.load(model_input)

# load testing data
Xtest = np.load(test_fname+'.npy')

# make prediction
yout = (model.predict(Xtest)>0.5).astype(np.int)

# write the prediction results to predict_output.csv
with open(predict_output+'.csv', 'wb') as outputfile:
	csv_output = csv.writer(outputfile)

	csv_output.writerow(['id', 'label'])

	for idx, value in enumerate(yout, 1):
		csv_output.writerow([idx, value])

