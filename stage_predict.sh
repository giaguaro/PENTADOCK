#!/bin/bash
source activate pentadock

test_path=$1
#features_path=$2
#calibration_path=$2
#calibration_features_path=$4
checkpoint_path=$2
preds_path=$3
evaluation_scores_path=$4
num_workers=$5
dropout_sampling_size=$6


mpnn_predict --test_path $test_path --checkpoint_path $checkpoint_path --evaluation_scores_path $evaluation_scores_path --preds_path $preds_path --uncertainty_method dropout --num_workers $num_workers --no_cache_mol --batch_size 264 --dropout_sampling_size $dropout_sampling_size --gpu 1 --empty_cache
