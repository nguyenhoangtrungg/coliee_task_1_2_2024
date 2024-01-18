from core.utilities import support_func
import pandas as pd

from core.constant import *

def create_weak_dataset(weak_dataset_path, min_len, max_len, model_tyoe = "bert"):
    weak_dataset = support_func.read_json(weak_dataset_path)
    fragment_list = []
    content_list = []
    label_list = []
    for weak_data in weak_dataset:
        fragment = weak_data[0]
        content = weak_data[1]
        label = weak_data[2]
        if model_tyoe == "t5":
            label = str((label == "1")).lower()
        len_fragment = len(fragment.split())
        len_content = len(content.split())
        if len_fragment < min_len or len_fragment > max_len:
            continue
        if len_content < min_len or len_content > max_len:
            continue
        fragment_list.append(fragment)
        content_list.append(content)
        label_list.append(label)
    return pd.DataFrame({"fragment": fragment_list, "content": content_list, "label": label_list})

if __name__ == "__main__":
    MIN_LEN = 20
    MAX_LEN = 100
    WEAK_DATASET_PATH = "data/weak_label/train_data.json"
    weak_dataset = create_weak_dataset(WEAK_DATASET_PATH, MIN_LEN, MAX_LEN)
    zero_label_count = weak_dataset["label"].value_counts()
    print(zero_label_count)
    print(len(weak_dataset))
    print(weak_dataset.head())