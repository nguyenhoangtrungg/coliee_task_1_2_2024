import math

from core.utilities import support_func
from core.model import metrics
from core.bm25.run_bm25 import run_create_csv_bm25
from core.model import model_dataset

import torch
import torch.nn as nn
import argparse
import transformers
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer

from constant import *

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

sigmoid_model = torch.nn.Sigmoid()
softmax_model = nn.Softmax(dim=1)


# DATA LOADING
train_df = run_create_csv_bm25(TRAINING_PATH, LABEL_PATH, CSV_TRAINING_DATA_PATH, "train", NEGATIVE_MODE, NEGATIVE_NUM)
valid_df = run_create_csv_bm25(TESTING_PATH, LABEL_PATH, CSV_TESTING_DATA_PATH, "test", NEGATIVE_MODE, NEGATIVE_NUM)

if FAST_DEV_RUN == 1:
    train_df = train_df[:40]
    valid_df = valid_df[:40]

train_dataset = model_dataset.MultilingualBertDataset(train_df["fragment"].tolist(), train_df["content"].tolist(), train_df["label"].tolist())
valid_dataset = model_dataset.MultilingualBertDataset(valid_df["fragment"].tolist(), valid_df["content"].tolist(), valid_df["label"].tolist())

# MODEL CREATER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(PRETRAIN_MODEL)
model = BertForSequenceClassification.from_pretrained(PRETRAIN_MODEL, num_labels=2)
model.to(device)

if FREEZE_MODE == 1:
    for param in model.bert.parameters():
        param.requires_grad = False
    for param in model.bert.embeddings.parameters():
        param.requires_grad = True

training_args = TrainingArguments(
    output_dir = '/kaggle/working/',
    num_train_epochs = N_EPOCH,
    per_device_train_batch_size = BATCH_SIZE,  
    per_device_eval_batch_size= 1,
    evaluation_strategy = "epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    disable_tqdm = False, 
    load_best_model_at_end=True
)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        label_rate = [1.0, float(NEGATIVE_NUM)]
        print("Rate:", label_rate)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = softmax_model(outputs.get("logits"))
#         print(labels, logits)
        loss_fct = nn.CrossEntropyLoss(weight=(torch.tensor(label_rate)).to("cuda"))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss
    

trainer = CustomTrainer(
    model=model,
    args=training_args,
    compute_metrics=metrics.compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset
)

print("Start training...")
trainer.train()