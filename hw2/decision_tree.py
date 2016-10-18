'''
decision_tree: Decision tree learning algorithm

===
Author:		Cheng-Shih, Wong
Stu. ID:	R04945028
Email:		mob5566[at]gmail.com

Provides:
	Decision tree model for binary classification with numerical features

===
Document

class dtree( max_depth=None, max_features=None, min_feature=1e-6, min_impurity=1e-6 )
	- build a decision tree
		
		max_depth: the maximum depth of tree (i.e. tree height), if is equal to None,
			then no height limits
		max_features: if it is not equal to None, it will
			sample the feature randomly when branching
		min_feature: the minimum gap between features to be distinguish
		min_impurity: the minimum gap between impurity to be distinguish
	
	fit( X, y ) - fit the training data (X, y)
		
		X: training data input features
		y: training data output labels
	
	predict( X ): y - given the input X, then predict the corresponding output y
		
		X: the features used to predict
		y: the prediction of X
		
'''

import numpy as np

INF = 1e9

class dtree(object):
	def __init__(self, max_depth=None, max_features=None,\
				min_feature=1e-6, min_impurity=1e-6):
		self.max_depth = max_depth
		self.max_features = max_features
		self.min_feat = min_feature
		self.min_impurity = min_impurity

		self.isLeaf = False
		self.val = None
		self.split_feat = None
		self.split_val = None
		self.childs = None
	
	def fit(self, X, y):

		data_num, feat_num = X.shape

		# generate random feature sampling mask
		mask = np.arange(feat_num)
		np.random.shuffle(mask)

		if self.max_features and self.max_features<feat_num:
			mask = mask[:self.max_features]

		# terminate 
		if impurity(y) < self.min_impurity or \
			np.all(X[:, mask].std(axis=0)<self.min_feat) or \
			(self.max_depth<1 if self.max_depth!=None else False) or \
			len(X)==1:

			self.isLeaf = True
			self.val = 1 if y.mean()>0.5 else 0
			return self

		minImp = INF
		minSmask = None

		for i in mask:
			interv = np.array(list(set(X[:,i])))

			if len(interv)==1: continue
			interv = np.array([(interv[k]+interv[k+1])/2 \
				for k in np.arange(len(interv)-1)])
			
			for val in interv:
				smask = X[:, i]<val

				imp = impurity(y[smask])*smask.sum() +\
					impurity(y[np.logical_not(smask)])*np.logical_not(smask).sum()

				if imp < minImp:
					minImp = imp
					self.split_feat = i
					self.split_val = val
					minSmask = smask

		if minSmask==None or np.all(minSmask):
			self.isLeaf = True
			self.val = 1 if y.mean()>0.5 else 0
			return self
		
		nextdep = self.max_depth-1 if self.max_depth else None
		self.childs = [\
			dtree(nextdep, self.max_features, self.min_feat, self.min_impurity),\
			dtree(nextdep, self.max_features, self.min_feat, self.min_impurity)]

		self.childs[0].fit(X[minSmask], y[minSmask])
		self.childs[1].fit(X[np.logical_not(minSmask)], y[np.logical_not(minSmask)])

		return self

	def predict(self, X):
		if self.isLeaf:
			return self.val
		else:
			smask = X[:, self.split_feat]<self.split_val
			y = np.zeros(len(X))
			y[smask] = self.childs[0].predict(X[smask])
			y[np.logical_not(smask)] = self.childs[1].predict(X[np.logical_not(smask)])
		return y

def impurity(y):
	mu_y = y.mean()
	return mu_y if mu_y < 0.5 else 1-mu_y
