import math
import os

from core.utilities import support_func
from core.model import metrics
from core.bm25.run_bm25 import run_create_csv_bm25
from core.model import t5_model_dataset
from core.model import infer_model
from core.weak_label import create_weak_dataset

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

from constant import *

import nltk
import evaluate
import numpy as np
nltk.download("punkt", quiet=True)
metric = evaluate.load("exact_match")

# DATA LOADING
if CHOOSE_WEAK == 0:
    train_df = run_create_csv_bm25(TRAINING_PATH, LABEL_PATH, CSV_TRAINING_DATA_PATH, "train", NEGATIVE_MODE, NEGATIVE_NUM, "t5")

elif CHOOSE_WEAK == 1:
    train_df = create_weak_dataset.create_weak_dataset(WEAK_DATASET_PATH, MIN_LEN, MAX_LEN, "t5")

valid_df = run_create_csv_bm25(TESTING_PATH, LABEL_PATH, CSV_TESTING_DATA_PATH, "test", NEGATIVE_MODE, NEGATIVE_NUM, "t5")

if FAST_DEV_RUN == "1":
    train_df = train_df[:40]
    valid_df = valid_df[:40]

print(valid_df["label"][:10])

train_dataset = t5_model_dataset.T5Dataset(train_df["fragment"].tolist(), train_df["content"].tolist(), train_df["label"].tolist())
valid_dataset = t5_model_dataset.T5Dataset(valid_df["fragment"].tolist(), valid_df["content"].tolist(), valid_df["label"].tolist())

# MODEL CREATER
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAIN_MODEL)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, model_max_length=512)
model.to(device)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    print("Preds: ", preds, "Labels: ", labels)

    # decode preds and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    print("Pred: ", decoded_preds, "Label: ", decoded_labels)

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    print("Pred: ", decoded_preds, "Label: ", decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return result

# data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
training_args = Seq2SeqTrainingArguments(
    output_dir="/kaggle/working/",
    evaluation_strategy = "epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate = LEARNING_RATE,
    num_train_epochs = N_EPOCH,
    per_device_train_batch_size = BATCH_SIZE, 
    per_device_eval_batch_size= 1,
    save_total_limit=5,
    predict_with_generate=True,
    disable_tqdm = False, 
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics
)

trainer.train()