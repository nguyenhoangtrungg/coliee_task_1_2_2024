import math
import os
from core.model import metrics
from core.bm25.run_bm25 import run_create_csv_bm25
from core.model import model_dataset
from core.weak_label import create_weak_dataset

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer

from constant import *

import wandb
os.environ["WANDB_API_KEY"] = 'd161976e53bc2384922e3ed5b3f8676c7b8a648d'

wandb.login()

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

sigmoid_model = torch.nn.Sigmoid()
softmax_model = nn.Softmax(dim=1)

# DATA LOADING
if CHOOSE_WEAK == 0:
    train_df = run_create_csv_bm25(TRAINING_PATH, TRAIN_LABEL_PATH, CSV_TRAINING_DATA_PATH, "train", NEGATIVE_MODE, NEGATIVE_NUM)

elif CHOOSE_WEAK == 1:
    train_df = create_weak_dataset.create_weak_dataset(WEAK_DATASET_PATH, CSV_WEAK_DATA_PATH, MIN_LEN, MAX_LEN, "bert", banlance=0)

valid_df = run_create_csv_bm25(TESTING_PATH, TEST_LABEL_PATH, CSV_TESTING_DATA_PATH, "test", NEGATIVE_MODE, NEGATIVE_NUM)

if FAST_DEV_RUN == "1":
    train_df = train_df[:40]
    valid_df = valid_df[:40]

train_dataset = model_dataset.MultilingualBertDataset(train_df["fragment"].tolist(), train_df["content"].tolist(), train_df["label"].tolist())
valid_dataset = model_dataset.MultilingualBertDataset(valid_df["fragment"].tolist(), valid_df["content"].tolist(), valid_df["label"].tolist())

# MODEL CREATER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(TOKENIZER)
model = BertForSequenceClassification.from_pretrained(PRETRAIN_MODEL, num_labels=2)
model.to(device)

if FREEZE_MODE == 1:
    for param in model.bert.parameters():
        param.requires_grad = False
    for param in model.bert.embeddings.parameters():
        param.requires_grad = True

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    do_train=True,
    learning_rate = LEARNING_RATE,
    num_train_epochs = N_EPOCH,
    per_device_train_batch_size = BATCH_SIZE,  
    per_device_eval_batch_size= BATCH_SIZE,
    evaluation_strategy="epoch",
    # logging_strategy="epoch",
    save_strategy="epoch",
    report_to="wandb",
    logging_steps=100,
    # evaluation_strategy="steps",
    logging_strategy="steps",
    disable_tqdm = False, 
    load_best_model_at_end=False,
    save_total_limit = 5
)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        label_rate = [1.0, float(W_LOSS)]
        # print("Rate:", label_rate)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = softmax_model(outputs.get("logits"))
#         print(labels, logits)
        loss_fct = nn.CrossEntropyLoss(weight=(torch.tensor(label_rate)).to(device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

if OPTIMIZER == "SGD":    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        compute_metrics=metrics.compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        optimizers = (torch.optim.SGD(model.parameters(), lr=LEARNING_RATE), None)
    )
else:
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        compute_metrics=metrics.compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

print("Start training...")
trainer.train()

# list_output_dir = os.listdir("/kaggle/working")
# for file in list_output_dir:
#     if "checkpoint" in file and "csv" not in file:
#         print("Found checkpoint:", file)
#         checkpoint_path = os.path.join("/kaggle/working", file)
#         df = infer_model.encode_csv(valid_df, 1, checkpoint_path)
#         support_func.write_csv(checkpoint_path + ".csv", df)
