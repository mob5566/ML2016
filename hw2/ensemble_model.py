'''
bagging_model: Ensemble meta algorithms

===
Author:		Cheng-Shih, Wong
Stu. ID:	R04945028
Email:		mob5566[at]gmail.com

Provides:
	1. Aggregate models to meta algorithm with bagging data with replacement
	2. Random Forest Classifier

===
Document

class bagging(modelType, modelParameters, modelNum=100, sampleNum=None)
	- The bagging meta algorithm built on input model type

		modelType: the aggregate model type to use
		modelParameter: the constructor parameters for the model type
		modelNum: the number of model to be aggregated
		sampleNum: the number of bootstrap sampled data 

class random_forest(treeNum=10, max_depth=None, max_features=None,
				 min_feature=1e-6, min_impurity=1e-6, sampleNum=None,
				 oob=True, scoring=lgr.accuracy):
	- The random forest algorithm built on decision tree learning for regression

		treeNum: the number of tree to be aggregated
		max_depth: the maximum depth of tree (i.e. tree height), if is equal to None,
			then no height limits
		max_features: if it is not equal to None, it will
			sample the feature randomly when brancing
		min_feature: the minimum gap between features to be distinguish
		min_impurity: the minimum gap between impurity to be distinguish
		sampleNum: the number of bootstrap sampled data 
		oob: if oob is True, then will calculate the out-of-box score of forest
		scoring: the scoring function of oob

'''

import numpy as np
import decision_tree as dt
import logreg_model as lgr

class bagging(object):
	def __init__(self, modelType, modelParameters, modelNum=100, sampleNum=None):
		self.modelType = modelType
		self.modelPara = modelParameters
		self.modelNum = modelNum
		self.sampleNum = sampleNum
	
	def fit(self, X, y):
		self.models = []

		sampleNum = self.sampleNum if self.sampleNum else len(X)

		# for each random trained models
		for i in np.arange(self.modelNum):

			# random data sampling
			mask = np.random.randint(0, len(X), sampleNum)

			self.models.append( self.modelType(*self.modelPara) )
			self.models[-1].fit(X[mask], y[mask])

	def predict(self, X):
		return sum([model.predict(X) for model in self.models])/self.modelNum

class random_forest(object):
	def __init__(self, treeNum=10, max_depth=None, max_features=None,\
				 min_feature=1e-6, min_impurity=1e-6, sampleNum=None,\
				 oob=True, scoring=lgr.accuracy):
		self.treeNum = treeNum
		self.max_depth = max_depth
		self.max_features = max_features
		self.min_feature = min_feature
		self.min_impurity = min_impurity
		self.sampleNum = sampleNum
		self.oob = oob
		self.scoring = scoring

	def fit(self, X, y):
		self.trees = []

		sampleNum = self.sampleNum if self.sampleNum else len(X)
		
		# set random seed
		np.random.seed(1428)

		self.oob_score = 0

		# for each random trained trees
		for i in np.arange(self.treeNum):

			# random data sampling
			mask = np.random.randint(0, len(X), sampleNum)

			# get the out-of-box validation set
			oobmask = np.ones(len(X), dtype=bool)
			oobmask[mask] = False

			# add the tree
			self.trees.append( dt.dtree(self.max_depth, self.max_features,\
							   self.min_feature, self.min_impurity) )

			# train teh decision tree
			self.trees[-1].fit(X[mask], y[mask])

			# calculate this decision tree score
			self.oob_score = self.oob_score+\
				self.scoring(self.trees[-1], X[oobmask], y[oobmask])

		self.oob_score = self.oob_score/self.treeNum

	def predict(self, X):
		return (sum([tree.predict(X) for tree in self.trees])/self.treeNum)>=0.5
