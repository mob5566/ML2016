'''
model_selection: model selection module 

===
Author:		Cheng-Shih, Wong
Stu. ID:	R04945028
Email:		mob5566[at]gmail.com

Provides:
	Cross validation and validation functions for model selection

===
Document

def cross_valid(model, X, y, errf, fold=10, shuffle=True) 
	- the cross validation model selection
		
		model: input model to be selected from
		X: the data set input features
		y: the data set output labels
		errf: the error function for evaluate performance
		fold: the number of sets to be folded
		shuffle: do random shuffle to the data before cross validation

=====

def valid(model, X, y, errf, sampleProb=0.2, shuffle=True):
	- the model selection by validation
		
		model: input model to be selected from
		X: the data set input features
		y: the data set output labels
		errf: the error function for evaluate performance
		sampleProb: the sampling probability of validation set

'''

import numpy as np

eps = 1e-8

def cross_valid(model, X, y, errf, fold=10, shuffle=True):
	
	X = X.copy()
	y = y.copy()
		
	data_num, feat_num = X.shape
	batchSize = np.floor(data_num/fold)

	# set random seed
	# np.random.seed(0)

	# random permute 
	if shuffle:
		data = np.random.permutation(np.insert(X, features_num, y, axis=1))
	
	X = data[:batchSize*fold, :-1].reshape(fold, batchSize, features_num)
	y = data[:batchSize*fold, -1].reshape(fold, batchSize)

	scores = np.zeros(fold)
	eins = np.zeros(fold)
			
	# cross validation
	for k in np.arange(fold):

		# get the training set mask
		mask = np.arange(fold)!=k

		tX = X[mask, :, :].reshape((fold-1)*batchSize, feat_num)
		ty = y[mask, :].reshape((fold-1)*batchSize)

		# train the model by training set
		model.fit(tX, ty)

		# evaluate by error function
		scores[k] = errf(model, X[k, :], y[k, :])
		eins[k] = errf(model, tX, ty)


	print 'Ein: ', eins.mean()
	print 'Scores: ', scores
	print 'Score: ', scores.mean()

	return scores, eins

def valid(model, X, y, errf, sampleProb=0.2, shuffle=True):
	
	X = X.copy()
	y = y.copy()
		
	data_num, feat_num = X.shape
	batchSize = np.floor(data_num/fold)

	# set random seed
	# np.random.seed(0)

	# select the training set and validation set
	trainmask = np.random.rand(data_num)>=sampleProb
	validmask = np.logical_not(trainmask)

	score = 0
	ein = 0

	# train the model by training set
	model.fit(X[trainmask], y[trainmask])

	# evaluate by error function
	score = errf(model, X[validmask], y[validmask])
	ein = errf(model, X[trainmask], y[trainmask])

	print 'Ein: ', ein 
	print 'Score: ', score

