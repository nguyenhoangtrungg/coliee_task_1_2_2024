from core.utilities import support_func
import os

def add_label(data, label):
    return data[label]

def create_single_paragraph_list(paragraph):
    score_list = []
    case_law_list = []
    for local_score in paragraph['top_score']:
        score_list.append(local_score['score'])
        case_law_list.append(local_score['in4'])
    
    score_list = support_func.min_max_scale(score_list)
    return score_list, case_law_list

def add_top_score(current, score_list, case_law_list):
    for i in range(len(case_law_list)):
        case_law = case_law_list[i]
        score = score_list[i]
        if case_law not in current:
            current[case_law] = score
        else:
            current[case_law] = max(current[case_law], score)
    
    return current

def create_paragraph_list(data, top_k=5):
    paragraph_list = {}
    for paragraph in data['paragraphs_top_score']:
        score_list, case_law_list = create_single_paragraph_list(paragraph)
        paragraph_list = add_top_score(paragraph_list, score_list, case_law_list)
    
    paragraph_list = sorted(paragraph_list.items(), key=lambda x: x[1], reverse=True)
    return paragraph_list[:top_k]

def run_single_create_paragraph_list(local_data, label, top_k=5):
    local_id = local_data["id"]
    local_top_case_law = create_paragraph_list(local_data, top_k)
    local_label = add_label(label, local_id)
    return {
        "id": local_id,
        "top_k": top_k,
        "top_case_law": local_top_case_law,
        "label": local_label
    }

def run_eval_single_case(case):
    label = case["label"]
    top_case_law = case["top_case_law"]
    predict = [x[0] for x in top_case_law]
    len_label = len(label)
    len_predict = len(predict)
    true_case = 0
    for local_predict in predict:
        if local_predict in label:
            true_case += 1
    
    return true_case, len_predict, len_label

def run_eval(data_path, label_path, top_k=5, log_path=""):
    data_list_path = support_func.get_list(data_path)
    label = support_func.read_json(label_path)
    total_true_case = 0
    total_predict = 0
    total_label = 0
    for data in data_list_path:
        local_data = support_func.read_json(data)
        result = run_single_create_paragraph_list(local_data, label, top_k)
        if log_path != "":
            local_save_path = os.path.join(log_path, result["id"] + ".json")
            support_func.write_json(local_save_path, result)
        true_case, len_predict, len_label = run_eval_single_case(result)
        total_true_case += true_case
        total_predict += len_predict
        total_label += len_label

    try:
        precision = total_true_case / total_predict
        recall = total_true_case / total_label
        f1_score = 2 * precision * recall / (precision + recall)
    except:
        precision = 0
        recall = 0
        f1_score = 0

    return {
        "top_k": top_k,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

if __name__ == '__main__':
    data_path = "data\es_output_label_v1"
    label_path = "data/task1_case_retrieval/task1_train_labels_2024.json"
    result = run_eval(data_path, label_path, 200, "data/log")
    print(result)