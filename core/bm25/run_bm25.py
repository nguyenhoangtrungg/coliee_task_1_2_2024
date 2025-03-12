from core.bm25 import create_bm25
from core.utilities import support_func

import os
import json

import numpy as np
import pandas as pd

from tqdm import tqdm


def create_case(folder_path):
    base_case = os.path.join(folder_path, "base_case.txt")
    entailed_fragment = os.path.join(folder_path, "entailed_fragment.txt")

    paragraph_path = os.path.join(folder_path, "paragraphs")

    # read txt file
    with open(base_case, "r", encoding="utf-8") as f:
        base_case = f.read()

    with open(entailed_fragment, "r", encoding="utf-8") as f:
        entailed_fragment = f.read()

    list_paragraphs = []
    paragraph_path = os.listdir(paragraph_path)
    for paragraph in paragraph_path:
        with open(
            os.path.join(folder_path, "paragraphs", paragraph), "r", encoding="utf-8"
        ) as f:
            list_paragraphs.append(paragraph + "#$%" + f.read())

    return base_case, entailed_fragment, list_paragraphs


def run_bm25(folder_path, label_list, topk=5):
    base_case, entailed_fragment, list_paragraphs = create_case(folder_path)
    topk = min(topk, len(list_paragraphs))
    list_paragraphs = support_func.list_preprocess(list_paragraphs)
    bm25_model, corpus = create_bm25.create_bm25_model(list_paragraphs)
    output = create_bm25.single_query_bm25(entailed_fragment, corpus, bm25_model, topk)
    # bm25 = create_bm25(list_paragraphs)
    return {
        "id": folder_path[-3:],
        "entailed_fragment": entailed_fragment,
        "label": label_list,
        "output": output,
    }


def create_single_csv_format(
    case, negative_mode="random", negative_num=5, model_type="bert"
):
    id_list = []
    fragment_list = []
    content_list = []
    name_list = []
    score_list = []
    label_list = []
    for content in case["output"]:
        if content["name"] in case["label"]:
            id_list.append(case["id"])
            fragment_list.append(case["entailed_fragment"])
            content_list.append(content["content"])
            name_list.append(content["name"])
            score_list.append(content["score"])
            if model_type == "t5":
                label_list.append("true")
            else:
                label_list.append(1)
    negative_num *= len(case["label"])
    current_negative_num = 0
    if negative_num == 0:
        negative_num = 100000

    if negative_mode == "hard":
        for content in case["output"]:
            if current_negative_num >= negative_num:
                break
            if content["name"] not in case["label"]:
                id_list.append(case["id"])
                fragment_list.append(case["entailed_fragment"])
                content_list.append(content["content"])
                name_list.append(content["name"])
                score_list.append(content["score"])
                if model_type == "t5":
                    label_list.append("false")
                else:
                    label_list.append(0)
                current_negative_num += 1

    elif negative_mode == "random":
        ran_id_list = []
        ran_fragment_list = []
        ran_content_list = []
        ran_name_list = []
        ran_score_list = []
        ran_label_list = []
        for content in case["output"]:
            if content["name"] not in case["label"]:
                ran_id_list.append(case["id"])
                ran_fragment_list.append(case["entailed_fragment"])
                ran_content_list.append(content["content"])
                ran_name_list.append(content["name"])
                ran_score_list.append(content["score"])
                if model_type == "t5":
                    ran_label_list.append("false")
                else:
                    ran_label_list.append(0)
        if len(ran_fragment_list) < negative_num:
            negative_num = len(ran_fragment_list)
        min_random = 0
        max_random = 15
        ran_id_list = ran_id_list[min_random:max_random]
        ran_fragment_list = ran_fragment_list[min_random:max_random]
        ran_content_list = ran_content_list[min_random:max_random]
        ran_name_list = ran_name_list[min_random:max_random]
        ran_score_list = ran_score_list[min_random:max_random]
        negative_num = min(negative_num, (max_random - min_random))
        ran_list = support_func.random_list(ran_content_list, negative_num)
        ran_list.sort()
        for i in ran_list:
            id_list.append(ran_id_list[i])
            fragment_list.append(ran_fragment_list[i])
            content_list.append(ran_content_list[i])
            name_list.append(ran_name_list[i])
            score_list.append(ran_score_list[i])
            label_list.append(ran_label_list[i])

    df = pd.DataFrame(
        {
            "id": id_list,
            "fragment": fragment_list,
            "content": content_list,
            "name": name_list,
            "score": score_list,
            "label": label_list,
        }
    )

    return df


def create_csv_format(
    case_list, model_mode, negative_mode="random", negative_num=5, model_type="bert"
):
    if model_mode == "test" or model_mode == "infer":
        negative_mode = "hard"
        negative_num = 1000
    total_df = pd.DataFrame()
    for i in tqdm(range(len(case_list))):
        case = case_list[i]
        df = create_single_csv_format(case, negative_mode, negative_num, model_type)
        total_df = pd.concat([total_df, df])
    return total_df


def run_create_csv_bm25(
    data_path,
    label_path,
    output_path,
    model_mode,
    negative_mode="random",
    negative_num=5,
    model_type="bert",
):
    data_list = os.listdir(data_path)
    if model_mode == "infer":
        label = {}
    else:
        label = support_func.read_json(label_path)
    output = []
    for i in tqdm(range(len(data_list))):
        data = data_list[i]
        if model_mode == "infer":
            label_list = []
        else:
            try:
                label_list = label[data]
            except:
                label_list = []
        output_bm25 = run_bm25(os.path.join(data_path, data), label_list, 100)
        output.append(output_bm25)

    df = create_csv_format(output, model_mode, negative_mode, negative_num, model_type)
    if model_mode == "train":
        df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    label_path = "resource/task2_train_labels_2024.json"
    folder_path = "resource/test"
    output_path = "resource/test_task2.csv"

    run_create_csv_bm25(
        folder_path, label_path, output_path, "train", "random", 5, "bert"
    )
