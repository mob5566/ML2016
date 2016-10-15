#! /bin/bash
#
# Author:	Cheng-Shih, Wong
# Stu. ID:	R04945028
# E-mail:	mob5566[at]gmail.com
#

if [ "$#" -ne 3 ]; then
	echo "Usage: test.sh model test_data.csv predict_output"
	exit
fi

python data_process.py $2
python test.py $1 $2 $3
