import pandas as pd
import numpy as np
from tqdm import tqdm

from core.utilities import support_func

def min_max_scale_by_id(df):
    df = df.copy()
    df["id"] = df["id"].astype(str)
    df["bert_score"] = df["bert_score"].astype(float)
    df["bert_score"] = df.groupby("id")["bert_score"].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    return df

def evaluate_threshold(df, w_bm25, w_bert, threshold):
    df = df.copy()
    # df["bert_score"] = df["bert_score"].astype(float)
    # df["score"] = df["score"].astype(float)
    df["ense_score"] = df["bert_score"] * w_bert + df["score"] * w_bm25
    df["ense_score"] = df.groupby("id")["ense_score"].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    n_predict = 0
    n_true = 0
    for i in range(len(df)):
        if df["ense_score"][i] >= threshold:
            n_predict += 1
            n_true += int(df["label"][i])
    
    n_label = sum(df["label"].tolist())
    
    try:
        precistion = n_true / n_predict
        recall = n_true / n_label
        f1_score = 2 * precistion * recall / (precistion + recall)
    except:
        precistion = 0
        recall = 0
        f1_score = 0
    return {
        "w_bm25": w_bm25,
        "w_bert": w_bert,
        "threshold": threshold,
        "precision": precistion,
        "recall": recall,
        "f1_score": f1_score,
    }

def run_ensemble(df, step):
    output_list = []
    currend_df = df.copy()
    for w_bm25 in tqdm(np.arange(0, 1 + step, step)):
        w_bert = 1 - w_bm25
        for threshold in np.arange(0, 1 + step, step):
            # print("w_bm25: {}, w_bert: {}, threshold: {}".format(w_bm25, w_bert, threshold))
            eval_result = evaluate_threshold(currend_df, w_bm25, w_bert, threshold)
            output_list.append(eval_result)

    output_df = pd.DataFrame(output_list)
    return output_df

if __name__ == "__main__":
    data_path = "data/task2_output/checkpoint-8740.csv"
    df = pd.read_csv(data_path)
    df = min_max_scale_by_id(df)
    df.to_csv("data/task2_output/checkpoint-8740_min_max.csv", index=False)
    step = 0.01
    df = run_ensemble(df, step)
    df.to_csv("data/task2_output/checkpoint-8740_ensemble.csv", index=False)