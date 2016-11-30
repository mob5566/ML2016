#
# Author:	Cheng-Shih, Wong
# Stu. ID:	R04945028
# Email:	mob5566[at]gmail.com or r04945028[at]ntu.edu.tw
#

from __future__ import print_function

import numpy as np
import cPickle as pickle
import collections
import os
import csv
import re
import itertools
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

import matplotlib.pyplot as plt

if len(os.sys.argv) != 3:
  print('Usage: python avg_cluster.py data_dir result.csv')
  os.sys.exit()

data_dir = os.sys.argv[1]
data_dir = data_dir if data_dir[-1]=='/' else data_dir+'/'
result_file = os.sys.argv[2]

# Count for the time comsuming
st = time.time()

print('Pre-processing...')

# Load the training data for clustering
print('TF-IDF start')
with open(data_dir+'title_StackOverflow.txt', 'rb') as f:
  data = f.readlines()
with open(data_dir+'docs.txt', 'rb') as f:
  docs = f.readlines()

word_pattern = r'[A-Z]+[a-z]*|[A-Z]*[a-z]+'
word_re = re.compile(word_pattern)

data = [' '.join(word_re.findall(line)) for line in data]
docs = [word_re.findall(line) for line in docs]
docs = [' '.join(line) for line in docs if len(line)>=7]

data_test = list(data)
data.extend(docs)

# TF-IDF
tfidf = TfidfVectorizer(max_df=0.5, min_df=2,
                        max_features=20000,
                        stop_words='english')

data = tfidf.fit_transform(data)
print('TF-IDF done')

# Latent Semantic Analysis by SVD dimensionality reduction
print('LSA start')
svd = TruncatedSVD(5)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(data)
X = tfidf.transform(data_test)
X = lsa.transform(X)
print('LSA done')

# Train the k-means model with extracted features
print('Training K-means model')
kmeans = KMeans(n_clusters=20, random_state=142813, n_jobs=-1).fit(X)
lb = kmeans.labels_

print('Training done')

# Plot the result
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

print(collections.Counter(lb))

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 1000
plot_mask = np.zeros(len(X), dtype=bool)
plot_mask[:plot_only] = True
plot_mask = np.random.permutation(plot_mask)
low_dim_data = tsne.fit_transform(X[plot_mask])
labels = lb[plot_mask]

plt.figure(figsize=(18, 18))
plt.scatter(low_dim_data[:,0], low_dim_data[:,1], c=labels, cmap=plt.cm.hsv)
plt.savefig('clusters.png')

# Load the test file and write the results
res = []

with open(data_dir+'check_index.csv', 'r') as f:
  csv_reader = csv.reader(f)

  csv_reader.next()
  for row in csv_reader:
    uid, a, b = row
    a = int(a)
    b = int(b)
    res.append([uid, 1 if lb[a]==lb[b] else 0])

with open(result_file, 'w') as f:
  csv_writer = csv.writer(f)
  csv_writer.writerow(['ID', 'Ans'])
  
  for row in res:
    csv_writer.writerow(row)

print('The title clustering cost', time.time()-st, 'seconds.')
