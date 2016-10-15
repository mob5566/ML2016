# /bin/bash

./train.sh spam_data/spam_train.csv mymodel
./test.sh mymodel.pkl spam_data/spam_test.csv myresults
