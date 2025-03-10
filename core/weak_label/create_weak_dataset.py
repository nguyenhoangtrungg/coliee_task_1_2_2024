import pandas as pd
from langdetect import detect
from core.utilities import support_func
from core.constant import *
from tqdm import tqdm
import os
import numpy as np


def create_weak_dataset(
    weak_dataset_path,
    csv_weak_output_path,
    min_len,
    max_len,
    model_type="bert",
    banlance=0,
):
    try:
        finally_df = pd.read_csv(csv_weak_output_path, index_col=False)
        label_list = finally_df["label"].tolist()
        for i in range(len(label_list)):
            label_list[i] = str(label_list[i]).lower()
        finally_df["label"] = label_list
        # meomeo
    except:
        weak_dataset = support_func.read_json(weak_dataset_path)
        fragment_list = []
        content_list = []
        label_list = []
        for i in tqdm(range(len(weak_dataset))):
            weak_data = weak_dataset[i]
            fragment = weak_data[0]
            content = weak_data[1]
            label = str(weak_data[2])
            if model_type == "t5":
                label = str((label == "1"))
                label = label.lower()
            len_fragment = len(fragment.split())
            len_content = len(content.split())
            if len_fragment < 10 or len_fragment > 150:
                continue
            if len_content < 10 or len_content > 200:
                continue
            # if len_content + len_fragment < min_len or len_content + len_fragment > max_len:
            #     continue
            # fragment_lang = detect(fragment)
            # content_lang = detect(content)
            # if fragment_lang != 'en' or content_lang != 'en':
            #     continue
            fragment_list.append(fragment)
            content_list.append(content)
            label_list.append(label)

        finally_df = pd.DataFrame(
            {"fragment": fragment_list, "content": content_list, "label": label_list}
        )
        finally_df = finally_df.sample(frac=1).reset_index(drop=True)
        finally_df.to_csv(csv_weak_output_path, index=False)
    if banlance == 1:
        true_df = finally_df[finally_df["label"] == "true"].copy()
        false_df = finally_df[finally_df["label"] == "false"].copy()
        min_len = min(len(true_df), len(false_df))
        true_sampled = true_df.sample(n=min_len, random_state=42)
        false_sampled = false_df.sample(n=min_len, random_state=42)
        final_sampled_df = (
            pd.concat([true_sampled, false_sampled])
            .sample(frac=1)
            .reset_index(drop=True)
        )
        return final_sampled_df
    else:
        return finally_df


if __name__ == "__main__":
    MIN_LEN = 100
    MAX_LEN = 500
    WEAK_DATASET_PATH = "/data2/cmdir/home/test01/minhnt/coliee_task_1_2_2024/resource/weak/train_data.json"
    weak_dataset = create_weak_dataset(
        WEAK_DATASET_PATH,
        "/data2/cmdir/home/test01/minhnt/coliee_task_1_2_2024/resource/weak_en.csv",
        MIN_LEN,
        MAX_LEN,
        model_type="t5",
        banlance=0
    )
    # weak_dataset = pd.read_csv("/data2/cmdir/home/test01/minhnt/coliee_task_1_2_2024/resource/weak.csv", index_col=False)
    zero_label_count = weak_dataset["label"].value_counts()
    print(zero_label_count)
    print(len(weak_dataset))
    print(weak_dataset.head(10))
