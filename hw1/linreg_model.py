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

class LinearRegression( LearningRate=1. ):
	The linear regression model.

	fit( X, y, maxIter=100, useL2R=False, L2R_lambda=1.,
		useSGD=False, batchSize=100, useAdagrad=False ) 
	- fit the training data (X, y)
		X: training data features X
		y: training data output y
		maxIter: the maximum gradient descent iterations
		useL2R: use L2 Regularization
		L2R_lambda: if use L2 Reg., the penalty value lambda
		useSGD: use Stochastic Gradient Descent
		batchSize: if use SGD, the batchSize is the size of SGD batch
		useAdagrad: use Adagrad in gradient descent
	
	predict( X ): y - given the input X, then predict the
	corresponding output y
		X: the features used to predict
		y: the prediction of X

	setEta( eta ) - set the learning rate eta
		eta: learning rate
'''

import numpy as np

class LinearRegression(object):
	def __init__(self, eta=1.):
		self.eta = eta

	def fit(self, X, y, maxIter=100, useL2R=False, L2R_lambda=1.,\
			useSGD=False, batchSize=100, useAdagrad=False):

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
		for i in np.arange(maxIter):
			
			if useSGD:
				rnd = np.random.randint(0, len(X)-batchSize)
				u = X[rnd:rnd+batchSize]
				v = y[rnd:rnd+batchSize]
			else:
				u = X
				v = y

			# calculate the gradient of square error with current w and b
			dw = np.dot(2*((np.dot(u, self._w)+self._b)-v), u)
			db = 2*(((np.dot(u, self._w)+self._b)-v).sum())

			# if use Adagrad
			if useAdagrad:
				acc_dw = acc_dw+dw**2
				acc_db = acc_db+db**2
				dw = dw/np.sqrt(acc_dw)
				db = db/np.sqrt(acc_db)
			
			# update the w and b
			self._w = self._w-self.eta*dw
			self._b = self._b-self.eta*db
	
	def predict(self, x):
		return np.dot(np.array(x), self._w)+self._b

	def setEta(self, eta):
		self.eta = eta
	
	def getW(self):
		return self._w

	def getB(self):
		return self._b
