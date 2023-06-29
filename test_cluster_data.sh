#!/bin/bash
#SBATCH --profile=<all|none|[energy[,|task[,|filesystem[,|network]]]]>
source activate pentadock


python cluster_data.py --input_file sec_new_test_data.csv --output_prefix outt --smiles_col smile
