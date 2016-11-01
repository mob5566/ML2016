#! /bin/bash
#
# Author:	Cheng-Shih, Wong
# Stu. ID:	R04945028
# Email:	mob5566[at]gmail.com or r04945028[at]ntu.edu.tw
#

if [ "$#" -ne 3 ]; then
	echo "Usage: test.sh data_dir model predict_output"
	exit
fi

python test.py $1 $2 $3
