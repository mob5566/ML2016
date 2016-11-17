#! /usr/bin/python
#
# Author:	Cheng-Shih, Wong
# Stu. ID:	R04945028
# Email:	mob5566[at]gmail.com or r04945028[at]ntu.edu.tw
#

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import MaxPooling2D, Convolution2D
from keras.optimizers import SGD
import numpy as np
import sys
import cPickle as pickle

if __name__ == '__main__':
	
	# set the channels dimension is at index 1
	from keras import backend as K
	K.set_image_dim_ordering('th')

	# training directory and model name
	if len(sys.argv) != 3:
		print 'Usage: python self_train.py data_dir model'
		sys.exit()
	data_dir = sys.argv[1]
	modelname = sys.argv[2]

	try:
		# load the labeled data 5000*3*32*32
		with open(data_dir+'all_label.p', 'rb') as infile:
			all_label = pickle.load(infile)
		all_label = np.array(all_label).reshape(5000, 3, 32, 32)

		# setting the labels
		labels = np.zeros(50000).reshape(5000, 10)
		for i in range(10):
			labels[i*500:(i+1)*500, i] = 1

		# load the unlabeled data 45000*3*32*32
		with open(data_dir+'all_unlabel.p', 'rb') as infile:
			all_unlabel = pickle.load(infile)
		all_unlabel = np.array(all_unlabel).reshape(45000, 3, 32, 32)
	except:
		print 'Error: data_dir not found or training data does not exist'
		sys.exit()
		
	# set self-training parameters
	train_data = all_label
	train_label = labels			
	remain_data = all_unlabel		# set all unlabeled data to be labeling
	threshold = 0.9					# threshold to add a new labeled data
	T = 10							# total number of self-training rounds

	for t in np.arange(T):

		# setup CNN model
		model = Sequential()

		# convolutional layers
		model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(3, 32, 32)))
		model.add(Activation('relu'))
		model.add(Convolution2D(32, 3, 3, border_mode='valid'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Convolution2D(64, 3, 3, border_mode='valid'))
		model.add(Activation('relu'))
		model.add(Convolution2D(64, 3, 3, border_mode='valid'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
	
		model.add(Flatten())
	
		# feed-forward layers
		model.add(Dense(128))
		model.add(Activation('relu'))

		model.add(Dense(128))
		model.add(Activation('relu'))
	
		model.add(Dense(10))
		model.add(Activation('softmax'))
	
		model.summary()

		model.compile(loss='categorical_crossentropy',\
					  optimizer='adam',\
					  metrics=['accuracy'])
	
		his = model.fit(train_data, train_label,\
						nb_epoch=10,\
						batch_size=100)

		if his.history['acc'][-1] < 0.5:
			continue

		prob = model.predict(remain_data)
		
		mask = prob.max(axis=1) >= threshold
		unmask = np.logical_not(mask)

		newdata = remain_data[mask]
		newlabel = (prob[mask]>=threshold).astype(np.int64)

		train_data = np.insert(train_data, len(train_data), newdata, axis=0)
		train_label = np.insert(train_label, len(train_label), newlabel, axis=0)
		remain_data = remain_data[unmask]

		if len(remain_data) == 0:
			break

	'''
	# save the self training data
	with open(data_dir+'st_data.p', 'wb') as file:
		pickle.dump(train_data, file)
	with open(data_dir+'st_label.p', 'wb') as file:
		pickle.dump(train_label, file)
	'''

	# setup CNN model
	model = Sequential()

	# convolutional layers
	model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(3, 32, 32)))
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 3, 3, border_mode='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Convolution2D(64, 3, 3, border_mode='valid'))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3, border_mode='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Flatten())

	# feed-forward layers
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(10))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',\
				  optimizer='adam',\
				  metrics=['accuracy'])
	
	model.fit(train_data, train_label,\
			  nb_epoch=100,\
			  batch_size=100)

	# save model to modelname
	model.save(modelname)

