#!/bin/bash

grep -v -Ff <(awk -F ',' '{print $2}' clustered_train.csv clustered_valid.csv clustered_test.csv | sort | uniq) $1 | cut -d ',' -f 1-2 > clustered_predict.csv

sed 's/,/ /g' clustered_predict.csv >> clustered_predict.smi
