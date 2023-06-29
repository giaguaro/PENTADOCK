#!/bin/bash
#SBATCH --profile=<all|none|[energy[,|task[,|filesystem[,|network[,|memory[,|gpu]]]]]]>


#example: $ ./script.sh -d /path/to/dir -o output_file -w 4 -j 8 -l library_file -p protein_file -x 0 -y 0 -z 0 

#sbatch pentadock.sh -e glide -d $SCHRODINGER -o output -w 45 -j 2 -l truncated_test_data.smi -p 3ckp_protein.pdb -i 2 -m ligprep -t map4 -x 29.414 -y 3.374 -z 10.476


usage() {
    echo "Usage: $0 -e ENGINE -d DIR -o OUTPUT -w WORKERS -j JOBS -l LIBRARY -p PROTEIN -i ITERATIONS -m METHOD -t TYPE -x CENTER_X -y CENTER_Y -z CENTER_Z" 1>&2
    exit 1
}

while getopts ":e:d:o:w:j:l:p:i:m:t:x:y:z:" opt; do
    case ${opt} in
        e )
            ENGINE=$OPTARG
            ;;
        d )
            DIR=$OPTARG
            ;;
        o )
            OUTPUT=$OPTARG
            ;;
        w )
            WORKERS=$OPTARG
            ;;
        j )
            JOBS=$OPTARG
            ;;
        l )
            LIBRARY=$OPTARG
            ;;
        p )
            PROTEIN=$OPTARG
            ;;
        i )
            ITERATIONS=$OPTARG
            ;;
        m)
            METHOD=$OPTARG
            ;;
        t)
            TYPE=$OPTARG
            ;;
        x )
            CENTER_X=$OPTARG
            ;;
        y )
            CENTER_Y=$OPTARG
            ;;
        z )
            CENTER_Z=$OPTARG
            ;;
        \? )
            usage
            ;;
        : )
            echo "Invalid option: $OPTARG requires an argument" 1>&2
            usage
            ;;
    esac
done


if [ -z "$DIR" ] || [ -z "$OUTPUT" ] || [ -z "$WORKERS" ] || [ -z "$JOBS" ] || [ -z "$LIBRARY" ] || [ -z "$PROTEIN" ] || [ -z "$ITERATIONS" ] || [ -z "$METHOD" ] || [ -z "$TYPE" ] || [ -z "$CENTER_X" ] || [ -z "$CENTER_Y" ] || [ -z "$CENTER_Z" ]; then
    usage
fi


BDIM_X=20
BDIM_Y=20
BDIM_Z=20


echo "Engine : $ENGINE" #gnina or glide
echo "Directory: $DIR" #HOME of docking engine
echo "Output: $OUTPUT" # Where to place output
echo "Workers: $WORKERS" # number of CPUs
echo "Jobs: $JOBS" # Exclusive for Schrodinger programs. For the glide and ligprep - be generous. Specify the max for glide. It will automatically multiply that by 3 for ligprep. Equivalent to tokens for Schrodinger. 
echo "Library: $LIBRARY" # ligand library can be SDF or CSV.
echo "Protein: $PROTEIN" # Ideally prepared protein
echo "Iterations: $ITERATIONS" # n_iteration for active learning. The more the more certainty for the top 5% compounds
echo "Method of library prep: $METHOD" #auto3d or ligprep
echo "Type of fingerprints: $TYPE" #morgan or map4
echo "BDIM_X: $BDIM_X"
echo "BDIM_Y: $BDIM_Y"
echo "BDIM_Z: $BDIM_Z"
echo "Center X: $CENTER_X"
echo "Center Y: $CENTER_Y"
echo "Center Z: $CENTER_Z"


home="/data/ml_programs/pentadock"
# for Auto3D purpose
export BABEL_LIBDIR='/data/ml_programs/openbabel/build/lib/'
#cd $home
if conda env list | grep -q pentadock; then
    source activate pentadock
else
    conda env create -f environment.yml
    source activate pentadock
fi

mkdir $OUTPUT
cp $LIBRARY $OUTPUT
cp $PROTEIN $OUTPUT
cp utilities/{*.py,*.sh} $OUTPUT
cp -r utilities/extra_libs $OUTPUT
cd $OUTPUT

echo "stage 1 - Validating mols..."
if [[ "${LIBRARY##*.}" == "sdf" ]]; then
    echo "Input ligand library has .sdf extension. This means thats the library will be assumed to have valid smiles. Skipping ligand validation..."
    # This is the smiles equivalent of the sdf file/
    python ../utilities/sdf2csv_names.py $LIBRARY ${LIBRARY%.*}.smi
    # get the first two coluumns and switch cols (switch cols are defined as smiles, followed by names)
    cat ${LIBRARY%.*}.smi | cut -d ' ' -f 1-2 >> ordered_${LIBRARY%.*}.smi
    mv ordered_${LIBRARY%.*}.smi ${LIBRARY%.*}.smi
    rm ordered_${LIBRARY%.*}.smi
    #awk '{print $2, $1}' ${LIBRARY%.*}.smi > switched_cols_$LIBRARY ${LIBRARY%.*}.smi
    sed 's/ /,/g' ${LIBRARY%.*}.smi >> comma_${LIBRARY%.*}.smi
    # get the IDs for conformer splitting later
    #cut -d"," -f 2 comma_${LIBRARY%.*}.smi >> names_${LIBRARY%.*}.smi 
    sdf_library=$LIBRARY
elif [[ "${LIBRARY##*.}" == "smi" || "${LIBRARY##*.}" == "csv" ]]; then
    echo "Input ligand library is in the .smi or .csv format. We need to prepare this library appropriately... "
    mv $LIBRARY ${LIBRARY%.*}.smi
    # make sure we have initialy space delimited
    sed -i 's/,/ /g' ${LIBRARY%.*}.smi
    sed 's/ /,/g' ${LIBRARY%.*}.smi >> comma_${LIBRARY%.*}.smi
    # Check that input has at least two columns 
    num_cols=$(head -n 1 ${LIBRARY%.*}.smi | tr ' ' '\n' | wc -l)
    if [ $num_cols -lt 2 ]; then
      echo "Error: The input SMI file must have at least two columns. The first is the name, the second is the smiles"
      exit 1
    fi

    # check the order of the columns
    awk '{if ($1 !~ /\(/) {print "First column must contain smiles, second column must contain names"; exit1}}' ${LIBRARY%.*}.smi
    # get switched cols
    awk '{print $2, $1}' ${LIBRARY%.*}.smi > switched_cols_${LIBRARY%.*}.smi
    #python tautomerize.py $LIBRARY tauto_${LIBRARY%.*}.smi
else
    echo "Input ligand library has unknown file extension. Ensure it is either sdf, smi, or csv file type"
fi

# Extract the names of the first two columns
smi_col=$(head -n 1 ${LIBRARY%.*}.smi | cut -d ' ' -f 1)
name_col=$(head -n 1 ${LIBRARY%.*}.smi | cut -d ' ' -f 2)
echo "The first two columns in the input SMI file are ${smi_col} and ${name_col}."


echo "stage 2 - Clustering and Splitting Mols..." 
python ../utilities/cluster_data.py --input_file ${LIBRARY%.*}.smi --n_clusters 200 --output_prefix clustered --smiles_col $smi_col
# a faster way to populate the predict corpus with names that do not exist in the train, valid, test clustered files - using bash.
bash make_prediction_corpus.sh comma_${LIBRARY%.*}.smi

# Note to self: i have already taken care of CSV to SMI conversion for the predict corpus with the make_prediction_corpus script. 

echo "stage 3 - Generating 3D conformers if necessary..."

if [ -n "${sdf_library}" ]; then
    echo "generating conformers for the training set ..."
    awk -v name_header=${name_col} -v smiles_header=${smi_col} 'BEGIN {FS = " "} { if (NR == 1) { for (i = 1; i <= NF; i++) { if ($i == name_header) { name_idx = i } if ($i == smiles_header) { smiles_idx = i } } } else { print $name_idx, $smiles_idx } }' clustered_train.smi >> smiles_names_train.smi
    cut -d ' ' -f 1 smiles_names_train.smi >> names_train.txt

    awk 'BEGIN{RS="\\$\\$\\$\\$"; ORS="$$$$"}
         (NR==FNR){a[$1]=$0; next}
         ($1 in a) { print a[$1] }' $sdf_library RS="\n" names_train.txt >> clustered_train.sdf

    sdf_train=clustered_train.sdf
    echo "generating conformers for the validation set ..."
    awk -v name_header=${name_col} -v smiles_header=${smi_col} 'BEGIN {FS = " "} { if (NR == 1) { for (i = 1; i <= NF; i++) { if ($i == name_header) { name_idx = i } if ($i == smiles_header) { smiles_idx = i } } } else { print $name_idx, $smiles_idx } }' clustered_valid.smi >> smiles_names_valid.smi
    cut -d ' ' -f 1 smiles_names_valid.smi >> names_valid.txt

    awk 'BEGIN{RS="\\$\\$\\$\\$"; ORS="$$$$"}
         (NR==FNR){a[$1]=$0; next}
         ($1 in a) { print a[$1] }' $sdf_library RS="\n" names_valid.txt >> clustered_valid.sdf
    sdf_valid=clustered_valid.sdf
    echo "generating conformers for the testing set ..."
    awk -v name_header=${name_col} -v smiles_header=${smi_col} 'BEGIN {FS = " "} { if (NR == 1) { for (i = 1; i <= NF; i++) { if ($i == name_header) { name_idx = i } if ($i == smiles_header) { smiles_idx = i } } } else { print $name_idx } }' clustered_test.smi >> smiles_names_test.txt
    cut -d ' ' -f 1 smiles_names_test.smi >> names_test.txt

    awk 'BEGIN{RS="\\$\\$\\$\\$"; ORS="$$$$"}
         (NR==FNR){a[$1]=$0; next}
         ($1 in a) { print a[$1] }' $sdf_library RS="\n" names_test.txt >> clustered_test.sdf
    sdf_test=clustered_test.sdf

else
    awk -v name_header=${name_col} -v smiles_header=${smi_col} 'BEGIN {FS = " "} { if (NR == 1) { for (i = 1; i <= NF; i++) { if ($i == name_header) { name_idx = i } if ($i == smiles_header) { smiles_idx = i } } } else { print $smiles_idx, $name_idx} }' clustered_train.smi >> smiles_names_train.smi
    echo "${smi_col},${name_col}" | cat - smiles_names_train.smi > tmp.txt && mv tmp.txt smiles_names_train.smi
    awk -v name_header=${name_col} -v smiles_header=${smi_col} 'BEGIN {FS = " "} { if (NR == 1) { for (i = 1; i <= NF; i++) { if ($i == name_header) { name_idx = i } if ($i == smiles_header) { smiles_idx = i } } } else { print $smiles_idx, $name_idx} }' clustered_valid.smi >> smiles_names_valid.smi
    echo "${smi_col},${name_col}" | cat - smiles_names_valid.smi > tmp.txt && mv tmp.txt smiles_names_valid.smi
    awk -v name_header=${name_col} -v smiles_header=${smi_col} 'BEGIN {FS = " "} { if (NR == 1) { for (i = 1; i <= NF; i++) { if ($i == name_header) { name_idx = i } if ($i == smiles_header) { smiles_idx = i } } } else { print $smiles_idx, $name_idx} }' clustered_test.smi >> smiles_names_test.smi
    echo "${smi_col},${name_col}" | cat - smiles_names_test.smi > tmp.txt && mv tmp.txt smiles_names_test.smi
    if [[ "$METHOD" == "auto3d" ]]; then
        echo "We are using Auto3D to generate 3D conformers"

        echo "generating conformers for the training set ..."
        sed -i 's/,/ /g' smiles_names_train.smi
        sed -i 's/_/U/g' smiles_names_train.smi
        sed -i '1{/^[^0-9]*$/d}' smiles_names_train.smi

        python ../utilities/Auto3D_pkg/auto3D.py "smiles_names_train.smi" --k 1 --max_confs 1 --mpi_np $WORKERS --optimizing_engine AIMNET --verbose True --job_name output_train_auto3d --use_gpu=False
        sdf_train=$(ls output_train_auto3d_*/*sdf)
        echo "$sdf_train"

        echo "generating conformers for the validation set ..."
        sed -i 's/,/ /g' smiles_names_valid.smi
        sed -i 's/_/U/g' smiles_names_valid.smi
        sed -i '1{/^[^0-9]*$/d}' smiles_names_valid.smi

        python ../utilities/Auto3D_pkg/auto3D.py "smiles_names_valid.smi" --k 1 --max_confs 1 --mpi_np $WORKERS --optimizing_engine AIMNET --verbose True --job_name output_valid_auto3d --use_gpu=False
        sdf_valid=$(ls output_valid_auto3d_*/*sdf)
        echo "$sdf_valid"

        echo "generating conformers for the testing set ..."
        sed -i 's/,/ /g' smiles_names_test.smi
        sed -i 's/_/U/g' smiles_names_test.smi
        sed -i '1{/^[^0-9]*$/d}' smiles_names_test.smi
        python ../utilities/Auto3D_pkg/auto3D.py "smiles_names_test.smi" --k 1 --max_confs 1 --mpi_np $WORKERS --optimizing_engine AIMNET --verbose True --job_name output_test_auto3d --use_gpu=False
        sdf_test=$(ls output_test_auto3d_*/*sdf)
        echo "$sdf_test"


    elif [[ "$METHOD" == "ligprep" ]]; then
        echo "We are using LigPrep to generate 3D conformers"

        echo "generating conformers for the training set ..."
        sed -i 's/,/ /g' smiles_names_train.smi
        ${DIR}/ligprep -ismi smiles_names_train.smi -osd clustered_train.sdf -WAIT -NJOBS $(( JOBS * 3 )) -t 1 -bff 16 -i 0 -g
        sdf_train=clustered_train.sdf

        echo "generating conformers for the validation set ..."
        sed -i 's/,/ /g' smiles_names_valid.smi
        ${DIR}/ligprep -ismi smiles_names_valid.smi -osd clustered_valid.sdf -WAIT -NJOBS $(( JOBS * 3 )) -t 1 -bff 16 -i 0 -g
        sdf_valid=clustered_valid.sdf

        echo "generating conformers for the testing set ..."
        sed -i 's/,/ /g' smiles_names_test.smi
        ${DIR}/ligprep -ismi smiles_names_test.smi -osd clustered_test.sdf -WAIT -NJOBS $(( JOBS * 3 )) -t 1 -bff 16 -i 0 -g
        sdf_test=clustered_test.sdf

    else
        echo "Valid options for ligand preparation is 'auto3d' or 'ligprep', exiting ..."
        exit 1
    fi
fi

# do some organization

#mkdir clustered_smi
#mkdir clustered_sdf


#mv clustered*smi clustered_smi

# MORE ORGANIZATION HERE

echo "Preparing for docking"
echo "Stage 4 - Docking "

function ligand_docking {
    input=$1
    sdf=$2

    if [[ "$ENGINE" == "gnina" ]]; then

        echo "Docking the $input set..."
        cp ../utilities/gnina.dpf ./
        cp ../utilities/run_gnina.sh ./

        # modify values in gnina.dpf using sed
        sed -i "s/center_x\s*=\s*/center_x = $CENTER_X/" gnina.dpf
        sed -i "s/center_y\s*=\s*/center_y = $CENTER_Y/" gnina.dpf
        sed -i "s/center_z\s*=\s*/center_z = $CENTER_Z/" gnina.dpf
        sed -i "s/size_x\s*=\s*/size_x = $BDIM_X/" gnina.dpf
        sed -i "s/size_y\s*=\s*/size_y = $BDIM_Y/" gnina.dpf
        sed -i "s/size_z\s*=\s*/size_z = $BDIM_Z/" gnina.dpf
        sed -i "s/cpu\s*=\s*/cpu = $WORKERS/" gnina.dpf

        out_prefix="docked"

        # modify input and output file names in run_gnina.sh using sed
        sed -i "s/-r\s*protein.pdb/-r $PROTEIN/" run_gnina.sh
        sed -i "s/-l\s*library.sdf/-l ${sdf}/" run_gnina.sh
        sed -i "s/-o\s*output.sdf/-o ${out_prefix}_${sdf}/" run_gnina.sh
        sed -i "s/--cpu\s*cpus/--cpu $WORKERS/" run_gnina.sh

        sbatch -wait run_gnina.sh

        python ../utilities/sdf2csv.py ${out_prefix}_${sdf} ${out_prefix}_${sdf%.*}.csv
        score_col="minimizedAffinity"
        awk -v name_header=${score_col} -v smiles_header=${smi_col} 'BEGIN {FS = ","} { if (NR == 1) { for (i = 1; i <= NF; i++) { if ($i == name_header) { name_idx = i } if ($i == smiles_header) { smiles_idx = i } } } else { print $smiles_idx, $name_idx } }' ${out_prefix}_${sdf%.*}.csv > docked_${input}_dock.csv
        #cut -d "," -f 1,3 ${out_prefix}_${sdf%.*}.csv > ${input}_dock.csv
    sed -i 's/ /,/g' docked_${input}_dock.csv
    echo "${smi_col},${score_col}" | cat - docked_${input}_dock.csv > tmp.txt && mv tmp.txt docked_${input}_dock.csv
 
    elif [[ "$ENGINE" == "glide" ]]; then

        echo  "Docking the ${input} set..."
        cp ../utilities/grid.in ./
        cp ../utilities/glide.in ./

        sed -i "s/GRID_CENTER/GRID_CENTER   $CENTER_X, $CENTER_Y, $CENTER_Z/" grid.in
        sed -i "s|RECEP_FILE|RECEP_FILE   ${PWD}/${PROTEIN%.*}.mae|g" grid.in

        sed -i "s|GRIDFILE|GRIDFILE ${PWD}/grid.zip|g" glide.in
        sed -i "s|LIGANDFILE|LIGANDFILE ${PWD}/$sdf|g" glide.in

        ${DIR}/utilities/prepwizard -NOJOBID -disulfides -mse -fillsidechains -fillloops -propka_pH 7 -rehtreat ${PROTEIN} ${PROTEIN%.*}.mae -f S-OPLS -minimize_adj_h
        ${DIR}/glide grid.in -NOJOBID -OVERWRITE -DISP ignore
        # MODIFY
        ${DIR}/glide -WAIT -NJOBS $JOBS -OVERWRITE -HOST schrogpu1:${JOBS} -JOBNAME ${input}_dock glide.in

        while [[ $(squeue --name ${input}_dock| wc -l) -gt 1 ]]; do echo "wait for freed resource"; sleep 30s; done

        mv *.sdfgz ${input}_dock.sdf.gz
        gunzip ${input}_dock.sdf.gz
#        python ../utilities/sdf2csv.py ${input}_dock.sdf glide_${input}_dock.csv
        score_col="r_i_docking_score"
        smi_col="SMILES"
        awk -v name_header=${score_col} -v smiles_header=${smi_col} 'BEGIN {FS = ","} { if (NR == 1) { for (i = 1; i <= NF; i++) { if ($i == name_header) { name_idx = i } if ($i == smiles_header) { smiles_idx = i } } } else { print $smiles_idx, $name_idx } }' ${input}_dock.csv > docked_${input}_dock.csv
        #cut -d "," -f 1-2 ${input}_dock.csv >> tmp.csv
        sed -i 's/ /,/g' docked_${input}_dock.csv
        echo "${smi_col},${score_col}" | cat - docked_${input}_dock.csv > tmp.txt && mv tmp.txt docked_${input}_dock.csv
    else
        echo "Valid options for ligand docking is 'gnina' or 'glide', exiting ..."
        exit 1
    fi


    # Return the score_col variable
    # Return the score_col variable as a string
    echo "$score_col"
}

# call function
score_col=$(ligand_docking training ${sdf_train})
#just making sure

if [[ "$ENGINE" == "glide" ]]; then score_col="r_i_docking_score"; elif [[ "$ENGINE" == "gnina" ]]; then score_col="minimizedAffinity";fi;

echo "Docking score column: $score_col"

# Extract the names of the first two columns
smi_col=$(head -n 1 docked_training_dock.csv | cut -d ',' -f 1)
#score_col=$(head -n 1 training_dock.csv | cut -d ' ' -f 2)
#echo "The first two columns in the input SMI file are ${smi_col} and ${na."

echo "DEPRECATED/ will not work/ SKIPPING stage 5: Generating features"
#mkdir featurized
## train set
#echo "Featurizing the training set"

#sbatch --wait ../utilities/featurize_training.sh docked_training_dock.csv featurized/feats_clustered_train.pkl $smi_col $TYPE $WORKERS

echo "Stage 6 - Hyerparamter optimization part 1 of 2"

echo "Now we do hyperparamter optimization on a cross-valdiated training set while validation and testing set are being docking" 
#data=$1
#data_feats=$2
#score=$3
#output=$4
#workers=$5

sbatch ../utilities/stage_hyperopt.sh docked_training_dock.csv $score_col optimize_out $WORKERS

echo "Stage 7 - Docking Validation and testing set and generating features"

echo "Docking validation .."
# call function
score_col=$(ligand_docking validation ${sdf_valid})
echo "Docking score column: $score_col"
if [[ "$ENGINE" == "glide" ]]; then score_col="r_i_docking_score"; elif [[ "$ENGINE" == "gnina" ]]; then score_col="minimizedAffinity";fi
#echo "Featurizing validation set"
## valid set
#sbatch --wait ../utilities/featurize_training.sh docked_validation_dock.csv featurized/feats_clustered_valid.pkl $smi_col $TYPE $WORKERS


echo "Dockign testing .."
# call function
score_col=$(ligand_docking testing ${sdf_test})
if [[ "$ENGINE" == "glide" ]]; then score_col="r_i_docking_score"; elif [[ "$ENGINE" == "gnina" ]]; then score_col="minimizedAffinity";fi
echo "Docking score column: $score_col"

#echo "Featurizing testing set"
## test set
#sbatch --wait ../utilities/featurize_training.sh docked_testing_dock.csv featurized/feats_clustered_test.pkl $smi_col $TYPE $WORKERS 

# $(( WORKERS / 3 ))

#data_path=$1
#features_path=$2
#separate_val_path=$3
#separate_val_features_path=$4
#separate_test_path=$5
#separate_test_features_path=$6
#target_columns=$7
#depth=$8
#dropout=$9
#linked_hidden_size=${10}
#prefix=${11}
#num_workers=${12}

while [[ $(squeue --name optimize | wc -l) -gt 1 ]]; do echo "wait for freed resource"; sleep 5s; done
echo "Stage 8 - Hyerparamter optimization part 2 of 2"


#make sure
mv docked_training_dock.csv training_dock.csv
mv docked_validation_dock.csv validation_dock.csv
mv docked_testing_dock.csv testing_dock.csv

# get the TOP three hyperparamter settings from the cross validation
output=$(grep -A 2 "Trial results" optimize_out/verbose.log | grep -v "Trial results" | awk '{print $1,$2,$3,$4, $5,$6,$7,$8,$9}' | sed 's/\n/ /g' | sed 's/}/}\n/g' | sort -k5 -n -r | head -n 3 | awk '{print $1,$2,$3,$4,$5,$6,$7,$8,$9}')

# Loop over the parameter sets and assign values to variables
IFS=$'\n'
words=("one" "two" "three")

i=0
for param in $(echo "$output"); do
    prefix="${words[$i]}"
    depth=$(echo "$param" | grep -o "depth': [0-9]*" | grep -o "[0-9]*")
    dropout=$(echo "$param" | grep -o "dropout': [0-9\.]*" | grep -o "[0-9\.]*")
    ffn_num_layers=$(echo "$param" | grep -o "ffn_num_layers': [0-9]*" | grep -o "[0-9]*")
    linked_hidden_size=$(echo "$param" | grep -o "linked_hidden_size': [0-9]*" | grep -o "[0-9]*")

    # Print the parameter values
    echo "depth: $depth, dropout: $dropout, ffn_num_layers: $ffn_num_layers, linked_hidden_size: $linked_hidden_size, prefix: $prefix"

    sbatch --wait ../utilities/stage_train_opt.sh training_dock.csv validation_dock.csv testing_dock.csv $score_col $depth $dropout $linked_hidden_size $ffn_num_layers "$prefix" $WORKERS

    # Increment the counter variable
    i=$((i+1))
done


best_model=$(for file in *_train_out/verbose.log; do
    r2=$(grep -o "Overall test r2 = [-0-9\.]*" "$file" | grep -o "[^ =]*$" | head -n 1)
    echo "$file: $r2"
done | sort -k 2 -nr | head -n 1| cut -d ":" -f 1)

depth=$(grep -o "depth': [0-9]*" $best_model | grep -o "[0-9]*")
dropout=$(grep -o "dropout': [0-9\.]*" $best_model | grep -o "[0-9\.]*")
linked_hidden_size=$(grep -o "ffn_hidden_size': [0-9]*" $best_model | grep -o "[0-9]*")
ffn_num_layers=$(grep -m 1 -o "ffn_num_layers': [0-9]*" $best_model | grep -o "[0-9]*")

echo " The best model has the following params - depth: $depth, dropout: $dropout, linked_hidden_size: $ffn_hidden_size, ffn_num_layers: $ffn_num_layers"

dir_best="${best_model%/*}"

#while [[ $(squeue --name featurize_pt1| wc -l) -gt 1 ]]; do echo "wait for freed resource"; sleep 5s; done

#smi_col=$(head -n 1 clustered_predict.smi | cut -d ' ' -f 1)
#echo "featurizing the prediction dataset"
#sbatch --wait ../utilities/featurize_testing.sh clustered_predict.smi featurized/feats_clustered_predict.pkl $smi_col $TYPE $WORKERS

#awk -v name_header=${name_col} -v smiles_header=${smi_col} 'BEGIN {FS = " "} { if (NR == 1) { for (i = 1; i <= NF; i++) { if ($i == name_header) { name_idx = i } if ($i == smiles_header) { smiles_idx = i } } } else { print $smiles_idx, $name_idx } }' clustered_predict.smi >> predict_corpus.csv

cp clustered_predict.csv predict_corpus.csv

smi_col=$(head -n 1 docked_training_dock.csv | cut -d ',' -f 1)
#echo "${smi_col},${name_col}" | cat - predict_corpus.csv > tmp.txt && mv tmp.txt predict_corpus.csv
#test_path=$1
#features_path=$2
#calibration_path=$3
#calibration_features_path=$4
#checkpoint_path=$5
#preds_path=$6
#evaluation_scores_path=$7
#num_workers=$8


sbatch --wait ../utilities/stage_train.sh training_dock.csv validation_dock.csv testing_dock.csv $score_col $depth $dropout $linked_hidden_size $ffn_num_layers iter_${iter} $WORKERS;

echo "Stage 9 - Prediction"
#mkdir predictions

#make sure
sed -i 's/ /,/g' predict_corpus.csv
#cut -d ',' -f 1 predict_corpus.csv > smiles_predict_corpus.csv

echo "Iteration 0 predictions commencing now..."

iter=0
split -l 400000 predict_corpus.csv chunked_predict_corpus_

for chunk in chunked_predict_corpus_; do
    # Add the header to the temp file
    if ! head -n 1 $chunk | grep -q "smiles"; then sed -i '1i smiles,ID' $chunk;fi
    # feats_clustered_predict_g_temp_chunk_19.pkl
    mkdir predictions_${chunk%.*}
    #python remove_outliers.py $pkl docking_score ${pkl%.*}_clean
    sbatch --wait ../utilities/stage_predict.sh ${chunk%.*} ${dir_best}/fold_0/model_0/model.pt predictions_${chunk%.*}/preds.csv predictions_${chunk%.*}/eval.csv $WORKERS 3
done;

mkdir iter_${iter}_predictions
#combine all chunk predictions
tail -q -n +2 predictions_*/preds.csv | cat <(head -n 1 predictions_${pkl%.*}/preds.csv) - > iter_${iter}_predictions/preds.csv
rm -r predictions_*

sort -t',' -k3n iter_${iter}_predictions/preds.csv > sorted_data.txt
num_lines=$(wc -l sorted_data.txt | awk '{print int($1 * 0.05)}')
# get the lowest 10% of docking scores
head -n $num_lines sorted_data.txt > lowest_5pct.txt
# sort by uncertainty descending
sort -t',' -k4nr lowest_5pct.txt > sorted_lowest_5pct_universe.txt
if ! head -n 1 sorted_lowest_5pct_universe.txt | grep -q "smiles"; then
    echo 'smiles,ID,docking_score,uncertainty' | cat - sorted_lowest_5pct_universe.txt > tmp2.csv && mv tmp2.csv sorted_lowest_5pct_universe.txt
fi

num_lines=$(wc -l sorted_lowest_5pct_universe.txt | awk '{print int($1 * 0.01)}')
head -n $num_lines sorted_lowest_5pct_universe.txt | cut -d',' -f1-2 > top_5pct_highest_uncertainty_iter_${iter}_pred.csv
awk -F',' 'FNR==NR{a[$2];next} $2 in a{print}' top_5pct_highest_uncertainty_iter_${iter}_pred.csv all_chunks.csv >  top_5pct_highest_uncertainty_iter_${iter}.csv
tail -n +2 top_5pct_highest_uncertainty_iter_${iter}.csv | cut -d, -f1-3 > tmp3.csv && mv tmp3.csv top_5pct_highest_uncertainty_iter_${iter}.csv
cut -d ',' -f 1-3 ./training_dock.csv > extracted_cols_training_dock.csv
cat extracted_cols_training_dock.csv top_5pct_highest_uncertainty_iter_${iter}.csv > augmented_clustered_train.csv

tr '\r' ' ' < augmented_clustered_train.csv >augmented_clustered_train_clean_iter_${iter}.csv
sed -i 's/ //g' augmented_clustered_train_clean_iter_${iter}.csv

cp top_5pct_highest_uncertainty_iter_${iter}.csv top_5pct_highest_uncertainty_iter_1.csv
cp augmented_clustered_train_clean_iter_${iter}.csv augmented_clustered_train_clean_iter_1.csv



echo "Stage 10 - Active Learning"
for iter in $(seq 1 $ITERATIONS); do
    # here we are assuming the preds start with a file of two cols
    sbatch --wait ../utilities/stage_train.sh augmented_clustered_train_clean_iter_${iter}.csv validation_dock.csv testing_dock.csv $score_col $depth $dropout $linked_hidden_size $ffn_num_layers iter_${iter} $WORKERS;

    for chunk in chunked_predict_corpus_; do
        # Add the header to the temp file
        if ! head -n 1 $chunk | grep -q "smiles"; then sed -i '1i smiles,ID' $chunk;fi
        # feats_clustered_predict_g_temp_chunk_19.pkl
        mkdir predictions_${chunk%.*}
        #python remove_outliers.py $pkl docking_score ${pkl%.*}_clean
        sbatch --wait ../utilities/stage_predict.sh ${chunk%.*} iter_${iter}_train_out/fold_0/model_0/model.pt predictions_${chunk%.*}/preds.csv predictions_${chunk%.*}/eval.csv $WORKERS 3
    done;

    #rm chunked_lowest_5pct_*
    mkdir iter_${iter}_predictions
    #combine all chunk predictions
    tail -q -n +2 predictions_*/preds.csv | cat <(head -n 1 predictions_${pkl%.*}/preds.csv) - > iter_${iter}_predictions/preds.csv
    rm -r predictions_*

    sort -t',' -k3n iter_${iter}_predictions/preds.csv > sorted_data.txt
    num_lines=$(wc -l sorted_data.txt | awk '{print int($1 * 0.05)}')
    # get the lowest 10% of docking scores
    head -n $num_lines sorted_data.txt > lowest_5pct.txt
    # sort by uncertainty descending
    sort -t',' -k4nr lowest_5pct.txt > sorted_lowest_5pct_universe.txt
    #num_lines=$(wc -l sorted_lowest_10pct.txt | awk '{print int($1 * 0.05)}')
    # get the top 5% uncertainty
    #head -n $num_lines sorted_lowest_10pct_universe.txt | cut -d',' -f1-4 > top_5pct_highest_uncertainty_iter_universe.csv
    if ! head -n 1 sorted_lowest_5pct_universe.txt | grep -q "smiles"; then
        echo 'smiles,ID,docking_score,uncertainty' | cat - sorted_lowest_5pct_universe.txt > tmp2.csv && mv tmp2.csv sorted_lowest_5pct_universe.txt
    fi

    num_lines=$(wc -l sorted_lowest_5pct_universe.txt | awk '{print int($1 * 0.01)}')
    head -n $num_lines sorted_lowest_5pct_universe.txt | cut -d',' -f1-2 > top_5pct_highest_uncertainty_iter_$((iter+1))_pred.csv

    ####echo "${smi_col},${score_col}" | cat - top_5pct_highest_uncertainty_iter_$((iter+1))_pred.csv > tmp2.csv && mv tmp2.csv top_5pct_highest_uncertainty_iter_$((iter+1)).csv

    # Sort the data by docking score in ascending order
#    sort -t $'\t' -k3n predictions/preds.csv > sorted_data.txt

    # Determine the number of lines representing the lowest 10% of docking scores
 #   num_lines=$(wc -l sorted_data.txt | awk '{print int($1 * 0.1)}')

    # Take the first num_lines lines of sorted_data.txt
  #  head -n $num_lines sorted_data.txt > lowest_10pct.txt

    # Sort the lowest_10pct.txt file by uncertainty in descending order
 #   sort -t $'\t' -k4nr lowest_10pct.txt > sorted_lowest_10pct.txt

    # Determine the number of lines representing the highest 5% of uncertainties
 #   num_lines=$(wc -l sorted_lowest_10pct.txt | awk '{print int($1 * 0.05)}')

    # Take the first num_lines lines of sorted_lowest_10pct.txt
 #   head -n $num_lines sorted_lowest_10pct.txt | cut -d',' -f1-2 > top_5pct_highest_uncertainty_iter_${iter}.smi

 #   echo "${smi_col},${score_col}" | cat - top_5pct_highest_uncertainty_iter_${iter}.smi > tmp2.csv && mv tmp2.csv top_5pct_highest_uncertainty_iter_${iter}.smi

    if [[ "$METHOD" == "auto3d" ]]; then
        echo "We are using Auto3D to generate 3D conformers"

        echo "generating conformers for the training set ..."
        sed 's/,/ /g' top_5pct_highest_uncertainty_iter_$((iter+1))_pred.csv >> top_5pct_highest_uncertainty_iter_$((iter+1))_pred.smi
        sed -i 's/_/U/g' top_5pct_highest_uncertainty_iter_$((iter+1))_pred.smi
        sed -i '1{/^[^0-9]*$/d}' top_5pct_highest_uncertainty_iter_$((iter+1))_pred.smi

        python ../utilities/Auto3D_pkg/auto3D.py "top_5pct_highest_uncertainty_iter_$((iter+1))_pred.smi" --k 1 --max_confs 1 --mpi_np $WORKERS --optimizing_engine AIMNET --verbose True --job_name output_train_auto3d --use_gpu=False
        sdf_augment=$(ls output_train_auto3d_*/*sdf)
        echo "$sdf_train"

    elif [[ "$METHOD" == "ligprep" ]]; then
        echo "We are using LigPrep to generate 3D conformers"
        echo "generating conformers for the augmentation datase ..."
        sed 's/,/ /g' op_5pct_highest_uncertainty_iter_$((iter+1))_pred.csv >> top_5pct_highest_uncertainty_iter_$((iter+1))_pred.smi
        ${DIR}/ligprep -ismi top_5pct_highest_uncertainty_iter_$((iter+1))_pred.smi -osd ligprep_of_augment.sdf  -NOJOBID -NJOBS $(( JOBS * 3 )) -t 1 -bff 16 -i 0 -g
        sdf_augment="ligprep_of_augment.sdf"
    fi

    # dock

    #$sdf_augment

    echo "Docking augment .."
    # call function
    score_col=$(ligand_docking augment ${sdf_augment})
    if [[ "$ENGINE" == "glide" ]]; then score_col="r_i_docking_score"; elif [[ "$ENGINE" == "gnina" ]]; then score_col="minimizedAffinity";fi
    echo "Docking score column: $score_col"

   # echo "Featurizing augmentation dataset"
    # test set
    smi_col=$(head -n 1 docked_augment_dock.csv | cut -d ',' -f 1)
   # sbatch --wait ../utilities/featurize_training.sh docked_augment_dock.csv featurized/feats_clustered_augment_iter_${iter}.pkl $smi_col $TYPE $WORKERS

    echo "Augmenting for iter $iter"
    # Concatenate two files with the same header, but remove the header from one of the files
    cat training_dock.csv <(tail -n +2 docked_augment_dock.csv) > augmented_clustered_train_clean_iter_$((iter+1)).csv


  #  python ../utilities/concat_feats.py featurized/feats_clustered_train.pkl featurized/feats_clustered_augment_iter_${iter}.pkl featurized/feats_combined_iter_${iter}.pkl;
   # wait;
 ##   sbatch --wait ../utilities/stage_train.sh combined_dock_iter_${iter}.csv featurized/feats_combined_iter_${iter}.pkl validation_dock.csv featurized/feats_clustered_valid.pkl testing_dock.csv featurized/feats_clustered_test.pkl $score_col $depth $dropout $linked_hidden_size $ffn_num_layers iter_${iter} $WORKERS;

    rm augment_dock.csv

   # sbatch --wait ../utilities/stage_predict.sh smiles_predict_corpus.csv featurized/feats_clustered_predict.pkl testing_dock.csv featurized/feats_clustered_test.pkl ${dir_best}/fold_0/model_0/model.pt predictions/preds.csv predictions/eval.csv $WORKERS
    # remove temporary files
    rm sorted_data.txt
    rm lowest_10pct.txt
    rm sorted_lowest_10pct.txt
    rm top_5pct_highest_uncertainty_iter_${iter}.smi


    echo "Iteration ${iter} completed"
done

echo "All iterations completed"

echo "End of deep docking pipeline"

#TODO EVALUATIONS


