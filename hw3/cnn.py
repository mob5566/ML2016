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
	data_dir = sys.argv[1]
	modelname = sys.argv[2]

	# load the labeled data 5000*3*32*32
	with open(data_dir+'/all_label.p', 'rb') as infile:
		all_label = pickle.load(infile)
	all_label = np.array(all_label).reshape(5000, 3, 32, 32)

	# setting the labels
	labels = np.zeros(50000).reshape(5000, 10)
	for i in range(10):
		labels[i*500:(i+1)*500, i] = 1

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

	model.compile(loss='categorical_crossentropy',\
				  optimizer='adam',\
				  metrics=['accuracy'])
	
	model.fit(all_label, labels,\
			  nb_epoch=100,\
			  batch_size=100)
	
	# save the model
	model.save(modelname)

