import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch.nn as nn
import torch
import os

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


from task1.constant import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pre_train = PRETRAIN_MODEL
model = AutoModelForSequenceClassification.from_pretrained(pre_train, num_labels = 2).to(device)
tokenizer = AutoTokenizer.from_pretrained(pre_train)

df = pd.read_csv(TRAINING_PATH)
df = df.dropna(how='any',axis=0)

if N_SAMPLES != -1:
    df = df.sample(frac=1).reset_index(drop=True)
    df = df[:N_SAMPLES]

train = df.drop(columns=['label'])
label = df['label']

from sklearn.model_selection import train_test_split
train_data, val_data, train_label, val_label = train_test_split(train, label, test_size=.2, random_state=42)
train_data = train_data.reset_index().drop('index',axis=1)
val_data = val_data.reset_index().drop('index',axis=1)
train_label = train_label.reset_index().drop('index',axis=1)
val_label = val_label.reset_index().drop('index',axis=1)

if FREEZE_MODE == "1":
    for name, param in model.named_parameters():
        if name.startswith("longformer.encoder"): # choose whatever you like here
            param.requires_grad = False
        

class LongFormerDataset(torch.utils.data.Dataset):
   
    def __init__(self, base_case, candidate, labels):
        self.labels = labels
        self.base_case = base_case
        self.candidate = candidate
        self.tokenizer = AutoTokenizer.from_pretrained(pre_train)

    def __len__(self):
        return len(self.labels)

    def tokenize_pair_text(self, text_1, text_2):
        return self.tokenizer(text_1, text_2, padding='max_length', truncation=True)

    def __getitem__(self, index):
        encodings = self.tokenize_pair_text(self.base_case[index], self.candidate[index])
        item = {key: torch.tensor(val) for key, val in encodings.items()}
        item['labels'] = torch.tensor(self.labels[index])
        return item
    

train_dataset = LongFormerDataset(train_data['case'], train_data['candidate'], train_label['label'])
val_dataset = LongFormerDataset(val_data['case'], val_data['candidate'], val_label['label'])

softmax_model = nn.Softmax(dim=1)

os.environ["WANDB_DISABLED"] = "true"
training_args = TrainingArguments(
    output_dir = '/kaggle/working/',
    num_train_epochs = N_EPOCH,
    per_device_train_batch_size = BATCH_SIZE,    
    per_device_eval_batch_size= 1,
    evaluation_strategy = "epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    disable_tqdm = False, 
    load_best_model_at_end=True,
    run_name = 'longformer-classification',
    learning_rate = LEARNING_RATE,
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall' : recall,
        'f1': f1
    }

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        label_rate = [1.0, W_LOSS]
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = softmax_model(outputs.get("logits"))
        loss_fct = nn.CrossEntropyLoss(weight=(torch.tensor(label_rate)).to(device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
trainer = CustomTrainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()