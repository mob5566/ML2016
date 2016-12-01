#!/bin/bash

if [ $# -ne 2 ]; then
  echo 'Usage: cluster.sh data_dir results.csv'
  exit
fi

python tfidf_cluster.py $1 $2
