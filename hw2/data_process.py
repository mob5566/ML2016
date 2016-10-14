#! /usr/bin/python

import numpy as np
import csv

###
### Process Training data
###

# get training data from train.csv
csv_input = csv.reader(open('spam_data/spam_train.csv', 'rb'))

data = []
for row in csv_input:
	data.append(row)

data = np.array(data)

# delete the header lines
data = data[:, 1:]

data = data.astype(np.float)

# save the trainable data to numpy file
np.save('spam_data/training_data.npy', data)

###
### Process Testing data
###

# get testing data from test_X.csv
csv_input = csv.reader(open('spam_data/spam_test.csv', 'rb'))

data = []

for row in csv_input:
	data.append(row)

data = np.array(data)

# delete the header columns
data = data[:, 1:]

data = data.astype(np.float)

# save the testable data from test_X.csv
np.save('spam_data/testing_data.npy', data)
