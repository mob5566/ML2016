#! /usr/bin/python

import numpy as np
import csv
import sys

###
### Process data
###

if len(sys.argv)<2 or\
	sys.argv[1][-4:]!='.csv':

	print 'In data_process.py:'
	print '\tUsage: python data_process.py data_name.csv'
	exit()
	
fname = sys.argv[1][:-4]

# get data from argument 
with open(sys.argv[1], 'rb') as infile:
	csv_input = csv.reader(infile)

	data = []
	for row in csv_input:
		data.append(row)
	
	data = np.array(data)

# delete the header collumns
data = data[:, 1:]

data = data.astype(np.float)

# save the usable data to numpy file as fname.npy
np.save(fname+'.npy', data)
