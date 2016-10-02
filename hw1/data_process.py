#! /usr/bin/python

import numpy as np
import csv

###
### Process Training data
###

# get training data from train.csv
csv_input = csv.reader(open('data/train.csv', 'rb'))

data = []
for row in csv_input:
	data.append(row)

data = np.array(data)

# delete the header lines
data = data[1:, :]
data = data[:, 3:]

# convert the 'NR' no rain string to 0
data[ data[:,:]=='NR' ] = '0'

data = data.astype(np.float)

# transform data to (month x hour per month x elements) = (12, 20*24, 18)
year_data = np.zeros(12*20*24*18).reshape(12, 20*24, 18)
mstep = 20*18
dstep = 18

for month in np.arange(12):
	for day in np.arange(20):
		startmonth = month*mstep;
		year_data[month, day*24:(day+1)*24, :] = data[startmonth+day*dstep : \
			startmonth+(day+1)*dstep, :].T

# transform the year representative data to trainable data
# trainable data is represented as
# (number of trainable data x ten-hour elements) = ((480-10+1)*12, 18*9+1)
training_data = []

for month in np.arange(12):
	for segment in np.arange(480-10+1):
		training_data.append(year_data[month, segment:segment+9, \
			:].flatten())
		training_data[-1] = \
		np.insert(training_data[-1], 18*9, year_data[month, segment+9, 9])
		
training_data = np.array(training_data)

# save the trainable data to numpy file
np.save('data/trainable.npy', training_data)

###
### Process Testing data
###

# get testing data from test_X.csv
csv_input = csv.reader(open('data/test_X.csv', 'rb'))

data = []

for row in csv_input:
	data.append(row)

data = np.array(data)

# delete the header columns
data = data[:, 2:]

# convert the 'NR' no rain string to 0
data[ data[:,:]=='NR' ] = '0'
data = data.astype(np.float)

testing_data = []

# transform data to nine-hour elements 18*9 features
for i in np.arange(240):
	testing_data.append( data[i*18:(i+1)*18, :].T.flatten() )

testing_data = np.array(testing_data)

# save the testable data from test_X.csv
np.save('data/testable.npy', testing_data)
