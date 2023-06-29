#!/bin/bash
source activate pentadock

data_path=$1
#features_path=$2
separate_val_path=$2
#separate_val_features_path=$4
separate_test_path=$3
#separate_test_features_path=$6
target_columns=$4
prefix=$5
num_workers=$6

echo "data_path: $data_path"
echo "features_path: $features_path"
echo "separate_val_path: $separate_val_path"
echo "separate_val_features_path: $separate_val_features_path"
echo "separate_test_path: $separate_test_path"
echo "separate_test_features_path: $separate_test_features_path"
echo "target_columns: $target_columns"
echo "depth: $depth"
echo "dropout: $dropout"
echo "linked_hidden_size: $linked_hidden_size"
echo "ffn_num_layers: $ffn_num_layers"
echo "prefix: $prefix"
echo "num_workers: $num_workers"


python ./MPNN/train.py --data_path $data_path --separate_val_path $separate_val_path --separate_test_path $separate_test_path --target_columns $target_columns --metric r2 --save_dir ${prefix}_train_out --seed 0 --pytorch_seed 99 --log_frequency 1 --dataset_type regression --num_workers $num_workers --loss_function mse --no_cache_mol --gpu 1


#python ./MPNN/train.py --data_path $data_path --features_path $features_path --separate_val_path $separate_val_path --separate_val_features_path $separate_val_features_path --separate_test_path $separate_test_path --separate_test_features_path $separate_test_features_path --target_columns $target_columns --metric r2 --save_dir ${prefix}_train_out --seed 0 --pytorch_seed 99 --log_frequency 1 --dataset_type regression --num_workers $num_workers --loss_function mse --no_cache_mol --gpu 1
