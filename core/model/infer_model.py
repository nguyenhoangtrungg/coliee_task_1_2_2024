from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import torch
import torch.nn as nn

from core.constant import *

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT).to(device)

softmax_model = nn.Softmax(dim=1)

def take_predict_label(logits):
    logits = softmax_model(logits)
    logits = logits.tolist()
    label_1_score = []
    for i in logits:
        label_1_score.append(i[1])
    return label_1_score

def model_encode(questions, articles, checkpoint_model="Default"):
    """
    Encode questions and articles
    """
    with torch.no_grad():
        encoded_input = tokenizer(
            questions,
            articles,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(device)
        if checkpoint_model == "Default":
            output = model(**encoded_input)
        else:
            output = checkpoint_model(**encoded_input)
        return take_predict_label(output.logits)
    
def model_batch_encode(questions, articles, batch_size, checkpoint_model="Default"):
    """
    Encode questions and articles in batch size
    """
    predicts = []
    # for i in tqdm(range(0, len(questions), batch_size)):
    for i in range(0, len(questions), batch_size):
        i_start = i
        i_end = i + batch_size
        local_questions = questions[i_start:i_end]
        local_articles = articles[i_start:i_end]
        if checkpoint_model == "Default":
            local_predict = model_encode(local_questions, local_articles)
        else:
            local_predict = model_encode(local_questions, local_articles, checkpoint_model)
        predicts.extend(local_predict)
    return predicts

def encode_csv(df, batch_size, checkpoint_model="Default"):
    """
    Encode csv file
    """
    questions = df["fragment"].tolist()
    articles = df["content"].tolist()
    if checkpoint_model == "Default":
        predicts = model_batch_encode(questions, articles, batch_size)
    else:
        local_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_model).to(device)
        predicts = model_batch_encode(questions, articles, batch_size, local_model)
    df["bert_score"] = predicts
    return df