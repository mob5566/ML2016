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

class linreg( maxIter=100, eta=1e-2, useL2R=False, L2R_lambda=1.,
		useSGD=False, batchSize=100,
		featureOrder=None, useFeatureScaling=False,
		, useAdagrad=False):
	- The linear regression model.

		maxIter: the maximum gradient descent iterations
		eta: the learning rate of gradient descent
		useL2R: use L2 Regularization
		L2R_lambda: if use L2 Reg., the penalty value lambda
		useSGD: use Stochastic Gradient Descent
		batchSize: if use SGD, the batchSize is the size of SGD batch
		featureOrder: if it's not equal to None, it transform data to featureOrder-space
		useFeatureScaling: scale features by (X - mu(X))/sigma(X)
		useAdagrad: use Adagrad in gradient descent

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

eps = 1e-8

class linreg(object):
	def __init__(self, maxIter=100, eta=1e-2,\
				useL2R=False, L2R_lambda=1.,\
				useSGD=False, batchSize=100,\
				featureOrder=None, useFeatureScaling=False,\
				useAdagrad=False):
		
		self.maxIter = maxIter
		self.eta = eta
		self.useL2R = useL2R
		self.L2R_lambda = L2R_lambda
		self.useSGD = useSGD
		self.batchSize = batchSize
		self.featureOrder = featureOrder
		self.useFS = useFeatureScaling
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

		oriX = X

		# use feature order transform
		if self.featureOrder:
			tX = X
			for i in np.arange(2, self.featureOrder+1):
				tX = np.append(tX, X**i, axis=1)
			X = tX

		# use feature scaling
		if self.useFS:
			self.xmean = X.mean(axis=0)
			self.xstd = X.std(axis=0)
			X = (X-self.xmean)/(self.xstd+eps)

		# set random seed
		# np.random.seed(14)

		# initialize weights w and bias b with zeros
		self._w = np.random.rand(X.shape[1])
		self._b = np.array(0)

		# accumulate delta
		acc_dw = np.zeros(X.shape[1])
		acc_db = np.array(0)

		self.hist_e = []

		# gradient descent
		for i in np.arange(self.maxIter):
			
			if self.useSGD:
				rmask = np.arange(len(X))
				np.random.shuffle(rmask)
				mX = X[rmask]
				my = y[rmask]

				for i in np.arange(0, len(X)-self.batchSize, self.batchSize):
					tX = mX[i:i+self.batchSize]
					ty = my[i:i+self.batchSize]
					self.gradientDescent(tX, ty)
				
			else:
				self.gradientDescent(X, y)

			self.hist_e.append(RMSE(self, oriX, y))
	
	def predict(self, X):
		if self.featureOrder:
			tX = X
			for i in np.arange(2, self.featureOrder+1):
				tX = np.append(tX, X**i, axis=1)
			X = tX
		if self.useFS: X = (X-self.xmean)/(self.xstd+eps)

		return np.dot(np.array(X), self._w)+self._b
	
	def gradientDescent(self, X, y):
		# calculate the gradient of square error with current w and b
		dw = np.dot(2*((np.dot(X, self._w)+self._b)-y), X)
		db = 2*(((np.dot(X, self._w)+self._b)-y).sum())

		# if use L2 Regularization
		if self.useL2R:
			dw = dw+2*self.L2R_lambda*self._w

		# if use Adagrad
		if self.useAdagrad:
			acc_dw = acc_dw+dw**2
			acc_db = acc_db+db**2

			dw = dw/np.sqrt(acc_dw+eps)
			db = db/np.sqrt(acc_db+eps)
			
		# update the w and b
		self._w = self._w-self.eta*dw
		self._b = self._b-self.eta*db
		

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

		# set random seed
		np.random.seed(0)

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

		# set random seed
		np.random.seed(14)
		
		data_num = len(self.X)
		features_num = len(self.X[0])
		trainmask = np.random.rand(data_num)>=self.vsp
		validmask = np.logical_not(trainmask)

		mrow = len(self.models)
		mcol = len(self.models[0])

		self.scores = np.zeros(mrow*mcol).reshape(mrow, mcol)
		self.ein = np.zeros(mrow*mcol).reshape(mrow, mcol)

		# for each input models
		for i in np.arange(mrow):
			for j in np.arange(mcol):

				# train the model by training set
				self.models[i][j].fit(self.X[trainmask, :], self.y[trainmask])

				# evaluate by error function
				self.scores[i, j] = self.errf(\
					self.models[i][j], self.X[validmask, :], self.y[validmask])

				self.ein[i, j] = self.errf(\
					self.models[i][j], self.X[trainmask, :], self.y[trainmask])

		print 'Ein:\n', self.ein 
		print 'Scores:\n', self.scores 

		self.bestmodel_r = np.argmin(self.scores)/mcol
		self.bestmodel_c = np.argmin(self.scores)%mcol

		self.bestmodel = self.models[self.bestmodel_r][self.bestmodel_c]
	
	def getBestModel(self):
		return self.bestmodel

	def getScores(self):
		return self.scores
