import math
import os

from core.utilities import support_func
from core.model import metrics
from core.bm25.run_bm25 import run_create_csv_bm25
from core.model import model_dataset, t5_model_dataset
from core.model import infer_model
from core.weak_label import create_weak_dataset

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer

from constant import *

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

sigmoid_model = torch.nn.Sigmoid()
softmax_model = nn.Softmax(dim=1)

dev_df = run_create_csv_bm25(TRAINING_PATH, TRAIN_LABEL_PATH, CSV_TRAINING_DATA_PATH, "test", NEGATIVE_MODE, NEGATIVE_NUM)

test_df = run_create_csv_bm25(TESTING_PATH, TEST_LABEL_PATH, CSV_TESTING_DATA_PATH, "infer", NEGATIVE_MODE, NEGATIVE_NUM)

# dev_dataset = model_dataset.MultilingualBertDataset(dev_df["fragment"].tolist(), dev_df["content"].tolist(), dev_df["label"].tolist())
# test_dataset = model_dataset.MultilingualBertDataset(test_df["fragment"].tolist(), test_df["content"].tolist(), test_df["label"].tolist())

# t5_dev_dataset = t5_model_dataset.T5Dataset(dev_df["fragment"].tolist(), dev_df["content"].tolist(), dev_df["label"].tolist())
# t5_test_dataset = t5_model_dataset.T5Dataset(test_df["fragment"].tolist(), test_df["content"].tolist(), test_df["label"].tolist())

checkpoint_path = PRETRAIN_MODEL
d_df = infer_model.encode_csv(dev_df, 1, checkpoint_path)
t_df = infer_model.encode_csv(test_df, 1, checkpoint_path)

dev_path = os.path.join(OUTPUT_DIR, "bert_test.csv")
test_path = os.path.join(OUTPUT_DIR, "bert_infer.csv")

support_func.write_csv(dev_path, d_df)
support_func.write_csv(test_path, t_df)
