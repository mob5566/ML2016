'''
logreg_model: Logistic regression for binary classification

===
Author:		Cheng-Shih, Wong
Stu. ID:	R04945028
Email:		mob5566[at]gmail.com

Provides:
	Logistic regression model by gradient descent method
	with cross-entropy error function
	and the L2 regularization for binary classification

===
Document

class logreg( maxIter=100, eta=1e-2, useL2R=False, L2R_lambda=1.,
		useSGD=False, batchSize=100,
		featureOrder=None, useFeatureScaling=False,
		, useAdagrad=False):
	- The logistic regression model.

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

'''

import numpy as np

eps = 1e-8

class logreg(object):
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
		
		X = np.array(X).copy()
		y = np.array(y).copy()

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

		# data size
		data_num, feat_num = X.shape

		# set random seed
		np.random.seed(14)

		# initialize weights w and bias b with zeros
		self._w = np.random.rand(feat_num)
		self._b = np.array(0)

		# accumulate delta
		self.acc_dw = np.zeros(feat_num)
		self.acc_db = np.array(0)

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

			self.hist_e.append(mismatch(self, oriX, y))
	
	def predict(self, X):
		X = np.array(X)
		if self.featureOrder:
			tX = X
			for i in np.arange(2, self.featureOrder+1):
				tX = np.append(tX, X**i, axis=1)
			X = tX
		if self.useFS: X = (X-self.xmean)/(self.xstd+eps)

		return sigmoid(np.dot(X, self._w)+self._b)
	
	def gradientDescent(self, X, y):
		# calculate the gradient of cross-entropy with current w and b
		dw = -np.dot((y-sigmoid(np.dot(X, self._w)+self._b)), X)
		db = -(y-sigmoid(np.dot(X, self._w)+self._b)).sum()

		# if use L2 Regularization
		if self.useL2R:
			dw = dw+2*self.L2R_lambda*self._w

		# if use Adagrad
		if self.useAdagrad:
			self.acc_dw = self.acc_dw+dw**2
			self.acc_db = self.acc_db+db**2

			dw = dw/np.sqrt(self.acc_dw+eps)
			db = db/np.sqrt(self.acc_db+eps)
			
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
	
def accuracy(model, X, y):
	return 1-mismatch(model, X, y)

def mismatch(model, X, y):
	yout = model.predict(X)>0.5
	return np.logical_xor(yout, y).mean()

def sigmoid(X):
	return 1/(1+np.exp(-X))
