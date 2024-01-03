WORKSPACE=$(pwd)
TRAINING_FILE="core\trainning.py"
MODE="train"
PRETRAINED_MODEL="bert-base-multilingual-cased"
FREEZE_MODE="0"
BATCH_SIZE="8"
NUM_EPOCHS="3"
USED_GPU="0"
TRAINING_PATH="data\task2_case_entailment\train"
TESTING_PATH="data\task2_case_entailment\dev"
LABEL_PATH="data\task2_case_entailment\task2_train_labels_2024.json"
CSV_TRAINING_DATA_PATH="data\task2_case_entailment\train.csv"
CSV_TESTING_DATA_PATH="data\task2_case_entailment\dev.csv"
NEGATIVE_MODE="hard"
NEGATIVE_NUM="1"
FAST_DEV_RUN="1"
W_LOSS="1.0"


PYTHONPATH=$WORKSPACE python $TRAINING_FILE \
    --mode $MODE \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_mode $FREEZE_MODE \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --used_gpu $USED_GPU \
    --training_path $TRAINING_PATH \
    --testing_path $TESTING_PATH \
    --label_path $LABEL_PATH \
    --csv_training_data_path $CSV_TRAINING_DATA_PATH \
    --csv_testing_data_path $CSV_TESTING_DATA_PATH \
    --negative_mode $NEGATIVE_MODE \
    --negative_num $NEGATIVE_NUM \
    --fast_dev_run $FAST_DEV_RUN \
    --w_loss $W_LOSS
```