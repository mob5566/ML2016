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

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models.word2vec import Word2Vec

import matplotlib.pyplot as plt

if len(os.sys.argv) != 3:
  print('Usage: python tfidf_cluster.py data_dir result.csv')
  os.sys.exit()

data_dir = os.sys.argv[1]
data_dir = data_dir if data_dir[-1]=='/' else data_dir+'/'
result_file = os.sys.argv[2]

# Count for the time comsuming
st = time.time()

print('Pre-processing...')

# Load the training data for clustering
with open(data_dir+'title_StackOverflow.txt', 'rb') as f:
  data = f.readlines()

split_pattern = r'[\!\?\.]'
split_re = re.compile(split_pattern)
word_pattern = r'[A-Z]+[a-z]*|[A-Z]*[a-z]+'
word_re = re.compile(word_pattern)

data = [map(str.lower, word_re.findall(line)) for line in data]

# Filter stop words
with open('stopwords.txt', 'rb') as f:
  stop_words = set()

  for line in f.readlines():
    map(stop_words.add, word_re.findall(line))

data = [[word for word in line if word not in stop_words] for line in data]

# Stemming
print('Stemming start')
from gensim.parsing.porter import PorterStemmer
stemmer = PorterStemmer()

data = [map(stemmer.stem, line) for line in data]
print('Stemming end')

# Phrase detection
print('Phrasing start')
from gensim.models.phrases import Phrases
from gensim.models.phrases import Phraser

# bigram = Phrases(data)
bigram = Phrases(data)
bigramer = Phraser(bigram)

# bigramer = Phraser.load('bigram.ph')

# bigramer.save('bigram.ph')
print('Phrasing end')

# TF-IDF
print('TF-IDF start')
data = [' '.join(line) for line in bigramer[data]]
feat_dim = 428

tfidf = TfidfVectorizer(stop_words='english', max_df=0.05, min_df=2, max_features=feat_dim)

X = tfidf.fit_transform(data)

# with open('sw', 'w') as f:
#   for word in tfidf.stop_words_:
#     f.write(word+'\n')
# with open('voc', 'w') as f:
#   for word in tfidf.vocabulary_:
#     f.write(word+'\n')

print('TF-IDF done')

# Train the k-means model with extracted features
print('Training K-means model')
init_cen = np.zeros(20*feat_dim).reshape(20, -1)
init_cen[0] = 0.1

np.random.seed(142813)

for i in range(feat_dim):
  np.random.shuffle(init_cen[:, i])

init_cen = np.insert(init_cen, init_cen.shape[0], np.zeros(feat_dim), axis=0)

kmeans = KMeans(n_clusters=21, n_init=100, max_iter=1000, n_jobs=-1).fit(X)
lb = kmeans.labels_

print('Training done')

cnt_lb = collections.Counter(lb)

'''
# Plot the result
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

X = X.todense()

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
'''

# Load the test file and write the results
print('Writing result to', result_file)
res = []

with open(data_dir+'check_index.csv', 'r') as f:
  csv_reader = csv.reader(f)
  no_lb = cnt_lb.most_common(1)[0][0]

  csv_reader.next()
  for row in csv_reader:
    uid, a, b = row
    a = int(a)
    b = int(b)
    res.append([uid, 1 if lb[a]!=no_lb and lb[b]!=no_lb and lb[a]==lb[b] else 0])

with open(result_file, 'w') as f:
  csv_writer = csv.writer(f)
  csv_writer.writerow(['ID', 'Ans'])
  
  for row in res:
    csv_writer.writerow(row)

print('The title clustering cost', time.time()-st, 'seconds.')
