#!/bin/bash

# stop whole script when CTRL+C
trap "exit" INT

# ===============================================
# Prepare project environment
# ===============================================

if [ -d "./venv" ]; then
    source ./venv/bin/activate
else
    echo "Python environment does not exist"
    exit
fi;

# ===============================================
# Prepare dataset names
# ===============================================

declare -a DATASETS=(
    "MACCROBAT2018"
    "MACCROBAT2020"
)

declare -a MODELS=(
    "gliner_large_bio-v0.2"
    "gliner_large_bio-v0.1"
    "gliner_large-v2.1"
    "gliner_medium-v2.1"
    "gliner_small-v2.1"
    "gliner_multi-v2.1"
)

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do

        python projects/gliner/scripts/pubmed/02_format_json_to_gliner_train.py \
            --input_file ./data/interim/${DATASET}/train.json \
            --output_file ./data/final/${DATASET}/train.gliner.json \
            --model_name urchade/${MODEL}

        echo "Training model ${MODEL} on dataset ${DATASET}"
        python projects/gliner/scripts/train_model.py \
            --data_train_file ./data/final/${DATASET}/train.gliner.json \
            --model_name_or_path urchade/${MODEL} \
            --model_output_dir ./models/${MODEL} \
            --num_train_epochs 3 \
            --batch_size 8 \
            --learning_rate 5e-6

        echo "Evaluating model ${MODEL} on dataset ${DATASET}"
        python projects/gliner/scripts/evaluate_model.py \
            --data_test_file ./data/final/${DATASET}/test.gliner.json \
            --output_file ./results/${DATASET}/${MODEL}.json \
            --model_name_or_path ./models/${MODEL} \
            --threshold 0.5
    done;
done;
