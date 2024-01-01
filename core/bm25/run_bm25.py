from core.bm25 import create_bm25
from core.utilities import support_func

import os
import json

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
        with open(os.path.join(folder_path, "paragraphs", paragraph), "r", encoding="utf-8") as f:
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
        "entailed_fragment": entailed_fragment,
        "label": label_list,
        "output": output,
    }

if __name__ == "__main__":
    label_path = "data/task2_case_entailment/task2_train_labels_2024.json"
    label_list = support_func.read_json(label_path)
    folder_path = "data/task2_case_entailment/task2_train_files_2024"
    folder_path_list = os.listdir(folder_path)
    output = []
    for folder in folder_path_list:
        label = label_list[folder]
        output_bm25 = run_bm25(os.path.join(folder_path, folder), label, 50)
        output.append(output_bm25)

    with open("data/task2_output/task2_train_bm25_2024.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)