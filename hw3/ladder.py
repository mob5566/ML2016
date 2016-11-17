#! /usr/bin/python
#
# Author:	Cheng-Shih, Wong
# Stu. ID:	R04945028
# Email:	mob5566[at]gmail.com or r04945028[at]ntu.edu.tw
#

from keras.models import Sequential
import keras.backend as K
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers import MaxPooling2D, Convolution2D, UpSampling2D, Flatten
from keras.layers import GaussianNoise, BatchNormalization, LeakyReLU
from keras.layers import GlobalAveragePooling2D, Lambda
from keras.models import Model, load_model
import matplotlib.pyplot as plt
import numpy as np
import sys
import cPickle as pickle

if __name__ == '__main__':
	
	# set the channels dimension is at index 1
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
		all_label = all_label.astype('float32')/255.

		# setting the labels
		labels = np.zeros(50000).reshape(5000, 10)
		for i in range(10):
			labels[i*500:(i+1)*500, i] = 1

		# load the unlabeled data 45000*3*32*32
		with open(data_dir+'all_unlabel.p', 'rb') as infile:
			all_unlabel = pickle.load(infile)
		all_unlabel = np.array(all_unlabel).reshape(45000, 3, 32, 32)
		all_unlabel = all_unlabel.astype('float32')/255.
	except:
		print 'Error: data_dir not found or training data does not exist'
		sys.exit()

	# initialization parameters
	batchSize = 50
	img_depth = 3
	img_width = 32
	img_height = 32
	gauss_sigma = 0.3
	train_data = np.insert(all_label, len(all_label), all_unlabel, axis=0)
	train_label = np.zeros((50000, 10))
	
	# setup ladder network

	#
	# encoder layers
	#

	## input layers
	input_layer = Input(batch_shape=(batchSize, img_depth, img_width, img_height))

	## layer 1: 3x3 Conv 96 BN LeackRelu
	x = Convolution2D(96, 3, 3, border_mode='same')
	xx = BatchNormalization(mode=2, axis=1)
	y = LeakyReLU()

	z_corr = GaussianNoise(gauss_sigma)(input_layer)
	zpre_corr = x(z_corr)
	z_corr = xx(zpre_corr)
	z_corr = GaussianNoise(gauss_sigma)(z_corr)
	h_corr = y(z_corr)

	z_cn = x(input_layer)
	z_cn = xx(z_cn)
	h_cn = y(z_cn)
	
	## layer 2: 3x3 Conv 96 BN LeackRelu
	x = Convolution2D(96, 3, 3, border_mode='same')
	xx = BatchNormalization(mode=2, axis=1)
	y = LeakyReLU()

	zpre_corr = x(h_corr)
	z_corr = xx(zpre_corr)
	z_corr = GaussianNoise(gauss_sigma)(z_corr)
	h_corr = y(z_corr)

	z_cn = x(h_cn)
	z_cn = xx(z_cn)
	h_cn = y(z_cn)

	## layer 3: 3x3 Conv 96 BN LeackRelu
	x = Convolution2D(96, 3, 3, border_mode='same')
	xx = BatchNormalization(mode=2, axis=1)
	y = LeakyReLU()

	zpre_corr = x(h_corr)
	z_corr = xx(zpre_corr)
	z_corr = GaussianNoise(gauss_sigma)(z_corr)
	h_corr = y(z_corr)

	z_cn = x(h_cn)
	z_cn = xx(z_cn)
	h_cn = y(z_cn)

	## layer 4: 2x2 Max-pool BN
	x = MaxPooling2D((2, 2), border_mode='same')
	xx = BatchNormalization(mode=2, axis=1)

	zpre_corr = x(h_corr)
	z_corr = xx(zpre_corr)
	z_corr = GaussianNoise(gauss_sigma)(z_corr)
	h_corr = z_corr

	z_cn = x(h_cn)
	z_cn = xx(z_cn)
	h_cn = z_cn

	## layer 5: 3x3 Conv 192 BN LeackRelu
	x = Convolution2D(192, 3, 3, border_mode='same')
	xx = BatchNormalization(mode=2, axis=1)
	y = LeakyReLU()

	zpre_corr = x(h_corr)
	z_corr = xx(zpre_corr)
	z_corr = GaussianNoise(gauss_sigma)(z_corr)
	h_corr = y(z_corr)

	z_cn = x(h_cn)
	z_cn = xx(z_cn)
	h_cn = y(z_cn)
	
	## layer 6: 3x3 Conv 192 BN LeackRelu
	x = Convolution2D(192, 3, 3, border_mode='same')
	xx = BatchNormalization(mode=2, axis=1)
	y = LeakyReLU()

	zpre_corr = x(h_corr)
	z_corr = xx(zpre_corr)
	z_corr = GaussianNoise(gauss_sigma)(z_corr)
	h_corr = y(z_corr)

	z_cn = x(h_cn)
	z_cn = xx(z_cn)
	h_cn = y(z_cn)

	## layer 7: 3x3 Conv 192 BN LeackRelu
	x = Convolution2D(192, 3, 3, border_mode='same')
	xx = BatchNormalization(mode=2, axis=1)
	y = LeakyReLU()

	zpre_corr = x(h_corr)
	z_corr = xx(zpre_corr)
	z_corr = GaussianNoise(gauss_sigma)(z_corr)
	h_corr = y(z_corr)

	z_cn = x(h_cn)
	z_cn = xx(z_cn)
	h_cn = y(z_cn)

	## layer 8: 2x2 Max-pool BN
	x = MaxPooling2D((2, 2), border_mode='same')
	xx = BatchNormalization(mode=2, axis=1)

	zpre_corr = x(h_corr)
	z_corr = xx(zpre_corr)
	z_corr = GaussianNoise(gauss_sigma)(z_corr)
	h_corr = z_corr

	z_cn = x(h_cn)
	z_cn = xx(z_cn)
	h_cn = z_cn

	## layer 9: 3x3 Conv 192 BN LeackRelu
	x = Convolution2D(192, 3, 3, border_mode='same')
	xx = BatchNormalization(mode=2, axis=1)
	y = LeakyReLU()

	zpre_corr = x(h_corr)
	z_corr = xx(zpre_corr)
	z_corr = GaussianNoise(gauss_sigma)(z_corr)
	h_corr = y(z_corr)

	z_cn = x(h_cn)
	z_cn = xx(z_cn)
	h_cn = y(z_cn)
	
	## layer 10: 1x1 Conv 192 BN LeackRelu
	x = Convolution2D(192, 1, 1, border_mode='same')
	xx = BatchNormalization(mode=2, axis=1)
	y = LeakyReLU()

	zpre_corr = x(h_corr)
	z_corr = xx(zpre_corr)
	z_corr = GaussianNoise(gauss_sigma)(z_corr)
	h_corr = y(z_corr)

	z_cn = x(h_cn)
	z_cn = xx(z_cn)
	h_cn = y(z_cn)

	## layer 11: 1x1 Conv 10 BN LeackRelu
	x = Convolution2D(10, 1, 1, border_mode='same')
	xx = BatchNormalization(mode=2, axis=1)
	y = LeakyReLU()

	zpre_corr = x(h_corr)
	mu = K.mean(zpre_corr, [2, 3], True)
	sig = K.std(zpre_corr, [2, 3], True)
	z_corr = xx(zpre_corr)
	z_corr = GaussianNoise(gauss_sigma)(z_corr)
	z_corrs = z_corr
	h_corr = y(z_corr)

	z_cn = x(h_cn)
	z_cn = xx(z_cn)
	z_cns = z_cn
	h_cn = y(z_cn)

	## layer 12: Global Mean-pool
	out_h_corr = GlobalAveragePooling2D()(h_corr)
	out_h_cn = GlobalAveragePooling2D()(h_cn)

	## output layer: 10-way softmax
	output_corr = Activation('softmax')(out_h_corr)
	output_cn = Activation('softmax')(out_h_cn)

	#
	# decoder layers
	#

	## Gaussian Denoise function from original paper
	def g_gaussdenoise(args):
		z_c, u = args
		size = K.int_shape(z_c)
		a1 = K.variable(np.random.random(size))
		a2 = K.variable(np.random.random(size))
		a3 = K.variable(np.random.random(size))
		a4 = K.variable(np.random.random(size))
		a5 = K.variable(np.random.random(size))
		a6 = K.variable(np.random.random(size))
		a7 = K.variable(np.random.random(size))
		a8 = K.variable(np.random.random(size))
		a9 = K.variable(np.random.random(size))
		a10 = K.variable(np.random.random(size))

		m = a1 * K.sigmoid(a2 * u + a3) + a4 * u + a5
		v = a6 * K.sigmoid(a7 * u + a8) + a9 * u + a10

		z_est = (z_c-m) * v + m

		return z_est

	## layer 1: 
	u = BatchNormalization(axis=1)(h_corr)

	z_est = Lambda(g_gaussdenoise, output_shape=(10,8,8,))([z_corrs, u])
	z_ests = (z_est-mu)/(sig+K.epsilon())

	## set the loss function for unsupervised learning
	def ladder_loss_unsupervise(x, x_est):
		decoder_loss = K.mean(K.sum(K.square(z_ests-z_cns), axis=[1, 2, 3]))

		return decoder_loss
	
	## set the loss function for supervised learning
	def ladder_loss_supervise(x, x_est):
		supervised_loss = -K.mean(K.sum(x*K.log(x_est), axis=1))

		return supervised_loss

	## set encoder model and predict model
   	encoder = Model(input_layer, output_corr)
	predictor = Model(input_layer, output_cn)

	# train unsupervised learning
	encoder.compile(optimizer='adam', loss=ladder_loss_unsupervise)
	# encoder.summary()

	encoder.fit(train_data, train_label,\
				nb_epoch=10,\
				batch_size=batchSize,\
				validation_split=0.2)

	# train supervised learning
	encoder.compile(optimizer='adam', loss=ladder_loss_supervise, metrics=['accuracy'])

	encoder.fit(all_label, labels,\
				  nb_epoch=150,\
				  batch_size=batchSize)

	predictor.compile(optimizer='adam', loss='categorical_crossentropy')
	
	predictor.save(modelname)
