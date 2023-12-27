from core.bm25 import create_bm25

import os

def create_case(folder_path):
    base_case = os.path.join(folder_path, "base_case.txt")
    entailed_fragment = os.path.join(folder_path, "entailed_fragment.txt")
    
    paragraph_path = os.path.join(folder_path, "paragraphs")

    # read txt file
    with open(base_case, "r") as f:
        base_case = f.read()
    
    with open(entailed_fragment, "r") as f:
        entailed_fragment = f.read()

    list_paragraphs = []
    paragraph_path = os.listdir(paragraph_path)
    for paragraph in paragraph_path:
        with open(os.path.join(folder_path, "paragraphs", paragraph), "r") as f:
            list_paragraphs.append(f.read())

    return base_case, entailed_fragment, list_paragraphs

if __name__ == "__main__":
    print(create_case("data\\task2_case_entailment\\task2_train_files_2024\\001"))