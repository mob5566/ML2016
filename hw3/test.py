#! /usr/bin/python
#
# Author:	Cheng-Shih, Wong
# Stu. ID:	R04945028
# Email:	mob5566[at]gmail.com or r04945028[at]ntu.edu.tw
#

from keras.models import load_model
import cPickle as pickle
import numpy as np
import csv
import sys

if __name__ == '__main__':

	# get arguments
	if len(sys.argv) != 4:
		print 'Usage: python test.py data_dir input_model prediction.csv'
		sys.exit()
	
	data_dir = sys.argv[1]
	model_name = sys.argv[2]
	result_name = sys.argv[3]

	# load test data
	try:
		with open(data_dir+'/test.p', 'rb') as infile:
			test_file = pickle.load(infile)
	except:
		print 'Error: data_dir not found or test.p does not exist'
		sys.exit()
	
	test_data = np.array(test_file['data']).reshape(-1, 3, 32, 32)
	test_id = test_file['ID']

	# load model
	try:
		model = load_model(model_name)
	except:
		print 'Error: input_model not found'
		sys.exit()

	# make prediction by test data
	res = np.argmax(model.predict(test_data, 50), axis=1)

	# write the result to result_name
	with open(result_name, 'wb') as file:
		output = csv.writer(file)

		output.writerow(['ID', 'class'])

		for i in np.arange(len(res)):
			output.writerow([test_id[i], res[i]])
