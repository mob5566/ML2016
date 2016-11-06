#! /bin/bash
#
# Author:	Cheng-Shih, Wong
# Stu. ID:	R04945028
# Email:	mob5566[at]gmail.com or r04945028[at]ntu.edu.tw
#

if [ "$#" -ne 2 ]; then
	echo "Usage: train.sh data_dir output_model"
	exit
fi

python self_train.py $1 $2
