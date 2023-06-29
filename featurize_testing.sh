#!/bin/bash
#SBATCH --job-name=featurize_pt2
#SBATCH --profile=<all|none|[energy[,|task[,|filesystem[,|network]]]]>
source activate pentadock


python make_feats_predict.py $1 $2 $3 $4 $5
