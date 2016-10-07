'''
bagging_model: Ensemble meta algorithms

===
Author:		Cheng-Shih, Wong
Stu. ID:	R04945028
Email:		mob5566[at]gmail.com

Provides:
	1. Aggregate models to meta algorithm with bagging data with replacement
	2. Random Forest Regression

===
Document

class bagging(modelType, modelParameters, modelNum=100, sampleProb=0.4)
	- The bagging meta algorithm built on input model type

		modelType: the aggregate model type to use
		modelParameter: the constructor parameters for the model type
		modelNum: the number of model to be aggregated
		sampleProb: the sampling probability of bagging data

class random_forest(treeType, treePara, treeNum=10, sampleProb=0.7)
	- The random forest algorithm built on decision tree learning for regression

		treeType: the aggregate tree type to use
		treePara: the constructor parameters for the tree type
		treeNum: the number of tree to be aggregated
		sampleProb: the sampling probability of bagging data
'''

import numpy as np

class bagging(object):
	def __init__(self, modelType, modelParameters, modelNum=100, sampleProb=0.4):
		self.modelType = modelType
		self.modelPara = modelParameters
		self.modelNum = modelNum
		self.sampleProb = sampleProb
	
	def fit(self, X, y):
		self.models = []

		sampleNum = self.sampleProb*len(X)

		# for each random trained models
		for i in np.arange(self.modelNum):

			# random data sampling
			mask = np.random.randint(0, len(X), sampleNum)

			self.models.append( self.modelType(*self.modelPara) )
			self.models[-1].fit(X[mask], y[mask])

	def predict(self, X):
		return sum([model.predict(X) for model in self.models])/self.modelNum

class random_forest(object):
	def __init__(self, treeType, treePara, treeNum=10, sampleProb=0.7):
		self.treeType = treeType
		self.treePara = treePara
		self.treeNum = treeNum
		self.sampleProb = sampleProb

	def fit(self, X, y):
		self.trees = []

		sampleNum = self.sampleProb*len(X)

		# for each random trained trees
		for i in np.arange(self.treeNum):

			# random data sampling
			mask = np.random.randint(0, len(X), sampleNum)

			self.trees.append( self.treeType(*self.treePara) )
			self.trees[-1].fit(X[mask], y[mask])

	def predict(self, X):
		return sum([tree.predict(X) for tree in self.trees])/self.treeNum
