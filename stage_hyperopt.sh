#!/bin/bash
#SBATCH --job-name=optimize
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=0

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12355
export WORLD_SIZE=2

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source activate pentadock
data=$1
#data_feats=$2
score=$2
output=$3
workers=$4

#python ./MPNN/hyperparameter_optimization.py --data_path outlier_clustered_50k_train.csv --num_iters 10 --split_type cv-no-test --num_folds 2 --target_columns docking_score --metric r2 --save_dir optimized_50k_run_renewed_no_morgan --save_smiles_splits --seed 0 --pytorch_seed 99 --log_frequency 1 --save_preds --dataset_type regression --num_workers 24 --epochs 15 --loss_function mse --no_cache_mol --gpu 1 --config_save_path best_configs.json

#python -m torch.distributed.launch --nproc_per_node=2 --use_env MPNN/hyperparameter_optimization.py
python ./MPNN/hyperparameter_optimization.py --data_path $data --num_iters 10 --split_type cv-no-test --num_folds 2 --target_columns $score --metric r2 --save_dir $output --save_smiles_splits --seed 0 --pytorch_seed 99 --log_frequency 1 --save_preds --dataset_type regression --num_workers $workers --loss_function mse --config_save_path best_configs.json --no_cache_mol --resume_experiment --gpu 1 --epochs 15
