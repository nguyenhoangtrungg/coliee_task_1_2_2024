import math
import os

from core.utilities import support_func
from core.bm25.run_bm25 import run_create_csv_bm25

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from constant import *
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def tokenize_pair_text(text_1, text_2):
    return tokenizer(text_1, text_2, max_length = 512, padding='max_length', truncation=True, return_tensors="pt").to(device)

def get_probability_token(fragment, content):
    tokenized = tokenize_pair_text(fragment, content)
    outputs = model.generate(
        **tokenized,
        max_new_tokens=10,
#         num_beams=4,
#         num_return_sequences=4,
        return_dict_in_generate=True,
        output_scores=True,
    )
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    input_length = 1 if model.config.is_encoder_decoder else outputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]
    output = []
    for tok, score in zip(generated_tokens[0], transition_scores[0]):
        
        local_case = {
            "token": tokenizer.decode(tok),
            "score": score.cpu().data.numpy(),
            "s_score":  np.exp(score.cpu().data.numpy()),
        }
        output.append(local_case)
    for local_case in output:
        if local_case["token"] == "true":
            return local_case["s_score"]
        if local_case["token"] == "false":
            local_case["s_score"] = 1 - local_case["s_score"]
            return local_case["s_score"]
    return score

def infer_csv(df):
    fragment = df["fragment"].tolist()
    content = df["content"].tolist()
    try:
        label = df["label"].tolist()
    except:
        label = []*len(fragment)
    score_list = []
    for i in tqdm(range(len(fragment))):
        score_list.append(get_probability_token(fragment[i], content[i]))
        if i == 100:
            break
    with open("t5_score.txt", "w") as f:
        for i in score_list:
            f.write(str(i) + "\n")
    df["t5_score"] = score_list
    return df


checkpoint_path = CHECKPOINT
model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, model_max_length=512)
model.to(device)

dev_df = run_create_csv_bm25(TRAINING_PATH, TRAIN_LABEL_PATH, CSV_TRAINING_DATA_PATH, "test", NEGATIVE_MODE, NEGATIVE_NUM)

test_df = run_create_csv_bm25(TESTING_PATH, TEST_LABEL_PATH, CSV_TESTING_DATA_PATH, "infer", NEGATIVE_MODE, NEGATIVE_NUM)

d_df = infer_csv(dev_df)
t_df = infer_csv(test_df)

dev_path = os.path.join(OUTPUT_DIR, "t5_dev.csv")
test_path = os.path.join(OUTPUT_DIR, "t5_test.csv")

support_func.write_csv(dev_path, d_df)
support_func.write_csv(test_path, t_df)
