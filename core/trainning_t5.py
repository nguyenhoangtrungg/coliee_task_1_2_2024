from core.bm25.run_bm25 import run_create_csv_bm25
from core.model import t5_model_dataset
from core.weak_label import create_weak_dataset

import torch
import torch.nn as nn
from transformers import  AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from constant import *
 
import nltk
import evaluate
import numpy as np

print(args)
nltk.download("punkt", quiet=True)
metric = evaluate.load("exact_match")

promt = "Recognizing entailment between a decision fragment and a relevant legal paragraph."

# DATA LOADING
if CHOOSE_WEAK == 0:
    train_df = run_create_csv_bm25(TRAINING_PATH, TRAIN_LABEL_PATH, CSV_TRAINING_DATA_PATH, "train", NEGATIVE_MODE, NEGATIVE_NUM, "t5")

elif CHOOSE_WEAK == 1:
    train_df = create_weak_dataset.create_weak_dataset(WEAK_DATASET_PATH, CSV_WEAK_DATA_PATH, MIN_LEN, MAX_LEN, "t5", banlance=0)

elif CHOOSE_WEAK == 2:
    train_df = create_weak_dataset.create_weak_dataset(WEAK_DATASET_PATH, CSV_WEAK_DATA_PATH, MIN_LEN, MAX_LEN, "t5", banlance=1)
valid_df = run_create_csv_bm25(TESTING_PATH, TEST_LABEL_PATH, CSV_TESTING_DATA_PATH, "test", NEGATIVE_MODE, NEGATIVE_NUM, "t5")

if FAST_DEV_RUN == "1":
    train_df = train_df[:1]
    valid_df = valid_df[:1]

# print(valid_df["label"][:10])

print("Train data len: ", len(train_df))
print("Train data fragment: ", train_df["fragment"].tolist()[:5])
print("Train data content: ", train_df["content"].tolist()[:5])
print("Train data label: ", train_df["label"].tolist()[:5])

print("Valid data len: ", len(valid_df))
print("Valid data fragment: ", valid_df["fragment"].tolist()[:5])
print("Valid data content: ", valid_df["content"].tolist()[:5])
print("Valid data label: ", valid_df["label"].tolist()[:5])


train_dataset = t5_model_dataset.T5Dataset(train_df["fragment"].tolist(), train_df["content"].tolist(), train_df["label"].tolist())
valid_dataset = t5_model_dataset.T5Dataset(valid_df["fragment"].tolist(), valid_df["content"].tolist(), valid_df["label"].tolist())

# print(train_dataset.__getitem__(0))
# print(valid_dataset.__getitem__(0))

# MODEL CREATER
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAIN_MODEL)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
model.to(device)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # print("Preds: ", preds, "Labels: ", labels)

    # decode preds and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # print("Pred: ", decoded_preds, "Label: ", decoded_labels)

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    print("Pred: ", decoded_preds[:10], "Label: ", decoded_labels[:10])
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)

    num_label = 0
    num_true_label = 0
    num_predict = 0
    error_rate = 0
    for i in range(len(decoded_preds)):
        pred = decoded_preds[i]
        label = decoded_labels[i]
        if label == 'true':
            num_label += 1
        if label == 'true' and pred == 'true':
            num_true_label += 1
        if pred == 'true':
            num_predict += 1

        if label not in ['true', 'false']:
            error_rate += 1
    precision = num_true_label / num_predict if num_predict > 0 else 0
    recall = num_true_label / num_label if num_label > 0 else 0
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except:
        f1 = 0
    result['f1'] = f1
    result['precision'] = precision
    result['recall'] = recall
    result['error_rate'] = error_rate
    
    return result


# data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
if IS_FP16 == 1:
    fp16_label = True
else:
    fp16_label = False
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate = LEARNING_RATE,
    num_train_epochs = N_EPOCH,
    per_device_train_batch_size = BATCH_SIZE, 
    per_device_eval_batch_size= BATCH_SIZE,
    save_total_limit=5,
    predict_with_generate=True,
    disable_tqdm = False,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    # # optim=OPTIM,
    # label_smoothing_factor=LABEL_SMOOTHING_FACTER,
    # fp16=FP16,
    # seed=42,
    load_best_model_at_end=False,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics
)

trainer.train()