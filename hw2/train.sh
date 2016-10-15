#! /bin/bash
#
# Author:	Cheng-Shih, Wong
# Stu. ID:	R04945028
# E-mail:	mob5566[at]gmail.com
#

if [ "$#" -ne 2 ]; then
	echo "Usage: train.sh training_data.csv output_model"
	exit
fi

python data_process.py $1
python train.py $1 $2
