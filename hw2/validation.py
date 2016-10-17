#! /usr/bin/python
#
# Author:	Cheng-Shih, Wong
# Stu. ID:	R04945028
# Email:	mob5566[at]gmail.com or r04945028[at]ntu.edu.tw
#

import numpy as np
import logreg_model as lgr
import model_selection as ms
import csv
import time
# import decision_tree as dt
# import ensemble_model as em

# load training data
training_data = np.load('spam_data/spam_train.npy')

X = training_data[:, :-1]
y = training_data[:, -1]

# feature trimming


# cross validation

print('\nCross validation...\n')

models = [
lgr.logreg(3000, 0.1, True, 0.1, True, 30, False, True),
lgr.logreg(2000, 0.05, True, 0.03, True, 30, False, True)
]

eins = np.zeros(len(models))
scores = np.zeros(len(models))

for i, md in enumerate(models):
	tstart = time.time()
	score, ein = ms.cross_valid(md, X, y, lgr.mismatch, 10, True)

	eins[i] = ein.mean()
	scores[i] = score.mean()

	print('Cross validation cost %.3f seconds!' % (time.time()-tstart))

# plot the learning curve
import matplotlib.pyplot as plt
for i, md in enumerate(models):
	md.fit(X, y)
	plt.plot(np.arange(len(md.hist_e)), md.hist_e, 'o', label=str(i)+'th model')
plt.legend(loc='upper right')
plt.show()


plt.plot( np.arange(len(models)), eins, 'rx-', label='Insample Error')
plt.plot( np.arange(len(models)), scores, 'go-', label='Scores')
plt.legend(loc='upper right')
plt.show()

