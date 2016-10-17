#! /usr/bin/python
#
# Author:	Cheng-Shih, Wong
# Stu. ID:	R04945028
# Email:	mob5566[at]gmail.com or r04945028[at]ntu.edu.tw
#

import numpy as np 
import logreg_model as lgr
import time

data = np.load('spam_data/spam_train.npy')

X = data[:, :-1]
y = data[:,  -1]

md = lgr.logreg(100000, 0.05, True, 0.03, True, 30, False, True)

stime = time.time()

md.fit(X, y)

print('Training cost %.3f seconds!' % (time.time()-stime))

print 'Ein:', lgr.mismatch(md, X, y)

import matplotlib.pyplot as plt
plt.plot( np.arange(len(md.hist_e)), md.hist_e )
plt.show()
