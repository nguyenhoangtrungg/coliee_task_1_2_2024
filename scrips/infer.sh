WORKSPACE=$(pwd)
TRAINING_FILE="core\infer.py"
MODE="train"
PRETRAINED_MODEL="bert-base-multilingual-cased"
CHECKPOINT_PATH=""
USED_GPU="0"

TESTING_PATH="data\task2_case_entailment\dev"
LABEL_PATH="data\task2_case_entailment\task2_train_labels_2024.json"
CSV_TESTING_DATA_PATH="data\task2_case_entailment\dev.csv"


PYTHONPATH=$WORKSPACE python $TRAINING_FILE \
    --mode $MODE \
    --pretrained_model $PRETRAINED_MODEL \
    --checkpoint_path $CHECKPOINT_PATH \
    --used_gpu $USED_GPU \
    --testing_path $TESTING_PATH \
    --label_path $LABEL_PATH \
    --csv_testing_data_path $CSV_TESTING_DATA_PATH
```