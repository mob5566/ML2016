'''
linreg_model: Liear Regression Model

===
Author:		Cheng-Shih, Wong
Stu. ID:	R04945028
Email:		mob5566[at]gmail.com

Provides:
	Linear regression model by gradient descent method
	and the L2 regularization for linear regression

===
Document

class LinearRegression( maxIter=100, eta=1., useL2R=False, L2R_lambda=1.,
		useSGD=False, batchSize=100,
		useSGDRandomSample=True, SGDRandSampProb=0.05,
		, useAdagrad=False):
	- The linear regression model.

		maxIter: the maximum gradient descent iterations
		eta: the learning rate of gradient descent
		useL2R: use L2 Regularization
		L2R_lambda: if use L2 Reg., the penalty value lambda
		useSGD: use Stochastic Gradient Descent
		batchSize: if use SGD, the batchSize is the size of SGD batch
		useAdagrad: use Adagrad in gradient descent
		useSGDRandomSample: if use SGD and use SGD random sample, it will sample training
			data randomly from data
		SGDRandSampleProb: the probability of sampling training data

	fit( X, y ) - fit the training data (X, y)

		X: training data input features
		y: training data output labels
	
	predict( X ): y - given the input X, then predict the
	corresponding output y

		X: the features used to predict
		y: the prediction of X

=====

def RMSE(model, X, y) - the root-mean-square deviation error
		
		model: the model to be evaluated
		X: the test data input features
		y: the test data output labels

=====

class cross_valid( models, X, y, errf=RMSE, fold=10 ) - the cross validation model selection
		
		models: input models to be selected from
		X: the data set input features
		y: the data set output labels
		errf: the error function for evaluate performance
		fold: the number of sets to be folded
	
	scores() - do cross validation on folded training data and select the best model

=====

class validation( models, X, y, errf=RMSE, validsetprob=0.4 ) 
	- the model selection by validation
		
		models: input models to be selected from
		X: the data set input features
		y: the data set output labels
		errf: the error function for evaluate performance
		validsetprob: the sampling probability of validation set

	scores() - do validation on random sampled training data and
		select the best model by validation set

'''

import numpy as np

class LinearRegression(object):
	def __init__(self, maxIter=100, eta=1., useL2R=False, L2R_lambda=1.,\
				useSGD=False, batchSize=100,\
				useSGDRandomSample=False, SGDRandSampProb=0.05,\
				useAdagrad=False):
		
		self.maxIter = maxIter
		self.eta = eta
		self.useL2R = useL2R
		self.L2R_lambda = L2R_lambda
		self.useSGD = useSGD
		self.batchSize = batchSize
		self.useSGDRS = useSGDRandomSample
		self.useSGDrsp = SGDRandSampProb
		self.useAdagrad = useAdagrad

	def fit(self, X, y):

		# check whether the training data is empty or not
		if len(X)<=0 or len(y)<=0:
			raise AssertionError
		
		# check whether the size of training data (X, y) is consistent or not
		if len(X) != len(y):
			raise AssertionError
		
		X = np.array(X)
		y = np.array(y)

		# initialize weights w and bias b with zeros
		self._w = np.random.rand(X.shape[1])
		self._b = np.array(0)

		# accumulate delta
		acc_dw = np.zeros(X.shape[1])
		acc_db = np.array(0)

		# gradient descent
		for i in np.arange(self.maxIter):
			
			if self.useSGD:
				if self.useSGDRS:
					rndmask = np.random.rand(len(X))<self.useSGDrsp
					u = X[rndmask]
					v = y[rndmask]
				else:
					rnd = np.random.randint(0, len(X)-self.batchSize)
					u = X[rnd:rnd+self.batchSize]
					v = y[rnd:rnd+self.batchSize]
			else:
				u = X
				v = y

			# calculate the gradient of square error with current w and b
			dw = np.dot(2*((np.dot(u, self._w)+self._b)-v), u)
			db = 2*(((np.dot(u, self._w)+self._b)-v).sum())

			# if use L2 Regularization
			if self.useL2R:
				dw = dw+2*self.L2R_lambda*self._w.sum()

			# if use Adagrad
			if self.useAdagrad:
				acc_dw = acc_dw+dw**2
				acc_db = acc_db+db**2
				dw = dw/np.sqrt(acc_dw)
				db = db/np.sqrt(acc_db)
			
			# update the w and b
			self._w = self._w-self.eta*dw
			self._b = self._b-self.eta*db

	
	def predict(self, x):
		return np.dot(np.array(x), self._w)+self._b
	

	def getEta(self):
		return self.eta

	def getMaxIter(self):
		return self.maxIter

	def getUseL2R(self):
		return self.useL2R
	
	def getL2RLambda(self):
		return self.L2R_lambda
	
	def getBatchSize(self):
		return self.batchSize
	
	def getUseAdagrad(self):
		return self.useAdagrad
	
	def getW(self):
		return self._w

	def getB(self):
		return self._b
	
	def setEta(self, eta):
		self.eta = eta

	def setMaxIter(self, maxIter):
		self.maxIter = maxIter

	def setUseL2R(self, useL2R):
		self.useL2R = bool(useL2R)
	
	def setL2RLambda(self, L2Rlambda):
		self.L2R_lambda = L2Rlambda
	
	def setBatchSize(self, batchSize):
		self.batchSize = batchSize
	
	def setUseAdagrad(self, useAdagrad):
		self.useAdagrad = bool(useAdagrad)
	
	def setW(self, w):
		self._w = w

	def setB(self, b):
		self._b = b

def RMSE(model, X, y):
	yout = model.predict(X)
	return np.sqrt(((yout-y)**2).sum()/len(X))
	
class cross_valid(object):
	def __init__(self, models, X, y, errf=RMSE, fold=10):
		self.models = models
		
		# check whether the size of X and y are the same
		if len(X) != len(y):
			raise AssertionError

		self.X = np.array(X)
		self.y = np.array(y)
		
		self.errf = errf
		self.fold = fold
	
	def scores(self):
		
		data_num = len(self.X)
		features_num = len(self.X[0])
		batchSize = np.floor(data_num/self.fold)

		# random permute 
		data = np.random.permutation(np.insert(self.X, features_num, self.y, axis=1))
		
		X = data[:batchSize*self.fold, :-1].reshape(self.fold, batchSize, features_num)
		y = data[:batchSize*self.fold, -1].reshape(self.fold, batchSize)

		mrow = len(self.models)
		mcol = len(self.models[0])

		self.scores = np.zeros(mrow*mcol).reshape(mrow, mcol)

		# for each input models
		for i in np.arange(mrow):
			for j in np.arange(mcol):
				
				# cross validation
				for k in np.arange(self.fold):

					# get the training set mask
					mask = np.arange(self.fold)!=k

					# train the model by training set

					self.models[i][j].fit(\
						X[mask, :, :].reshape((self.fold-1)*batchSize, features_num),\
						y[mask, :].reshape((self.fold-1)*batchSize))

					# evaluate by error function
					self.scores[i, j] = self.scores[i, j]+\
						self.errf(self.models[i][j], X[k, :], y[k, :])

				self.scores[i, j] = self.scores[i, j]/self.fold

		print( self.scores )

		self.bestmodel_r = np.argmin(self.scores)/mcol
		self.bestmodel_c = np.argmin(self.scores)%mcol
		self.bestmodel = self.models[self.bestmodel_r][self.bestmodel_c]
	
	def getBestModel(self):
		return self.bestmodel

	def getScores(self):
		return self.scores

class validation(object):
	def __init__(self, models, X, y, errf=RMSE, validsetprob=0.4):
		self.models = models
		
		# check whether the size of X and y are the same
		if len(X) != len(y):
			raise AssertionError

		self.X = np.array(X)
		self.y = np.array(y)
		
		self.errf = errf
		self.vsp = validsetprob
	
	def scores(self):
		
		data_num = len(self.X)
		features_num = len(self.X[0])
		trainmask = np.random.rand(data_num)>=self.vsp
		validmask = np.logical_not(trainmask)

		mrow = len(self.models)
		mcol = len(self.models[0])

		self.scores = np.zeros(mrow*mcol).reshape(mrow, mcol)

		# for each input models
		for i in np.arange(mrow):
			for j in np.arange(mcol):

				# train the model by training set
				self.models[i][j].fit(self.X[trainmask, :], self.y[trainmask])

				# evaluate by error function
				self.scores[i, j] = self.errf(\
					self.models[i][j], self.X[validmask, :], self.y[validmask])

		print( self.scores )

		self.bestmodel_r = np.argmin(self.scores)/mcol
		self.bestmodel_c = np.argmin(self.scores)%mcol

		self.bestmodel = self.models[self.bestmodel_r][self.bestmodel_c]
	
	def getBestModel(self):
		return self.bestmodel

	def getScores(self):
		return self.scores
