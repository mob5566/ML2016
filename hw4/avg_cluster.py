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

if len(os.sys.argv) != 3:
  print('Usage: python avg_cluster.py data_dir result.csv')
  os.sys.exit()

data_dir = os.sys.argv[1]
data_dir = data_dir if data_dir[-1]=='/' else data_dir+'/'
result_file = os.sys.argv[2]

# Count for the time comsuming
st = time.time()

print('Pre-processing...')

# Load the embedding matrix and dictionarys
embeddings = np.load('em.npy')

with open('w2n.pkl', 'r') as f:
  w2n = pickle.load(f)

with open('n2w.pkl', 'r') as f:
  n2w = pickle.load(f)

# Load the training data for clustering
with open(data_dir+'title_StackOverflow.txt', 'r') as f:
  data = f.readlines()

# Filter the valid words by regular expression
word_pattern = r'[A-Z]+[a-z\d]*|[A-Z]*[a-z\d]+'
word_re = re.compile(word_pattern)

words = map(word_re.findall, data)
words = [[word.lower() for word in line] for line in words]

del data

# Load stop words from stopwords.txt
with open('data/stopwords.txt', 'rb') as f:
  stopwords = set((''.join([c if c.isalnum() else ' ' for c in f.read()])).split())

words = [[word for word in line if word not in stopwords] for line in words]

# Transform the titles into feature space by embedding matrix
title_cnt = [collections.Counter(line) for line in words]
doc_cnt = collections.Counter(itertools.chain.from_iterable(
                                [cnt.keys() for cnt in title_cnt]))

idf = {}

for word in doc_cnt.keys():
  idf[word] = np.log(len(words))-np.log(doc_cnt[word])

X = np.zeros(len(title_cnt)*embeddings.shape[1]).reshape(len(title_cnt), -1)

for i, cnt in enumerate(title_cnt):
  cnts = float(sum(cnt.values()))

  for word in cnt.keys():
    if word in w2n:
      X[i] += embeddings[w2n[word]]*(cnt[word]/cnts)*idf[word]

# Train the k-means model with extracted features
print('Training K-means model...')
kmeans = KMeans(n_clusters=20, random_state=142813, n_jobs=-1).fit(X)
lb = kmeans.labels_

print('Training done!')

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
