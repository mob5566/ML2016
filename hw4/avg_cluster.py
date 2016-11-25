#
# Author:	Cheng-Shih, Wong
# Stu. ID:	R04945028
# Email:	mob5566[at]gmail.com or r04945028[at]ntu.edu.tw
#

from __future__ import print_function

import numpy as np
import cPickle as pickle
import os
import csv

from sklearn.cluster import KMeans

if len(os.sys.argv) != 3:
  print('Usage: python avg_cluster.py data_dir result.csv')
  os.sys.exit()

data_dir = os.sys.argv[1]
data_dir = data_dir if data_dir[-1]=='/' else data_dir+'/'
result_file = os.sys.argv[2]

# Load the embedding matrix and dictionarys
embeddings = np.load('em.npy')

with open('w2n.pkl', 'r') as f:
  w2n = pickle.load(f)

with open('n2w.pkl', 'r') as f:
  n2w = pickle.load(f)

# Load the training data for clustering
with open(data_dir+'title_StackOverflow.txt', 'r') as f:
  data = [line.lower() for line in f.readlines()]

# Transform the titles into feature space by embedding matrix
with open('data/stopwords.txt', 'r') as f:
  stopwords = set(''.join([c.lower() if c.isalnum() else ' ' for c in f.read()]).split())

keywords = [word for word in w2n.keys() if word not in stopwords]

def getfeatures(line):
  global keywords
  global embeddings
  
  ret = []
  for word in keywords:
    tl = line
    pos = tl.find(word)
    while pos >= 0 and len(tl)>0:
      if tl==line:
        print(word)
      ret.append(embeddings[w2n[word]])
      tl = tl[pos+len(word):]
      pos = tl.find(word)

  assert len(ret)>0
  return np.array(ret).mean(axis=0)
  
X = np.array([getfeatures(line) for line in data])

# Train the k-means model with extracted features
kmeans = KMeans(n_clusters=20, random_state=142813, n_jobs=-1).fit(X)

# Load the test file and write the results
res = []

with open(data_dir+'check_index.csv', 'r') as f:
  csv_reader = csv.reader(f)

  csv_reader.next()
  for row in csv_reader:
    uid, a, b = row
    a = int(a)
    b = int(b)
    res.append([uid, 1 if kmeans.labels_[a]==kmeans.labels_[b] else 0])

with open(result_file, 'w') as f:
  csv_writer = csv.writer(f)
  csv_writer.writerow(['ID', 'Ans'])
  
  for row in res:
    csv_writer.writerow(row)
