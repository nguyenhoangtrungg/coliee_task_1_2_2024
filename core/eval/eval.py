from core.utilities import support_func
import pandas as pd

def csv_evaluate(df, score_type, threshold):
    predict = df[score_type].tolist()
    label = df["label"].tolist()

    num_predict = 0
    num_true = 0
    true_case = 0

    for i in range(len(predict)):
        if predict[i] >= threshold:
            num_predict += 1
            true_case += int(label[i])
        if label[i] == 1:
            num_true += int(predict[i])

    if num_predict != 0:
        precision = true_case / num_predict
    else:
        precision = 0
    
    if num_true != 0:
        recall = true_case / num_true
    else:
        recall = 0
    
    if precision + recall != 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0
    
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def evaluate(predict_list, threshold):
    num_predict = 0
    num_true = 0
    true_case = 0

    for predict in predict_list:
        for i in range(len(predict["output"])):
            output = predict["output"][i]
            if output["score"] < threshold:
                continue
            num_predict += 1
            num_true += len(predict["label"])
            if output["name"] in predict["label"]:
                true_case += 1
        
    if num_predict != 0:
        precision = true_case / num_predict
    else:
        precision = 0
    
    if num_true != 0:
        recall = true_case / num_true
    else:
        recall = 0
    
    if precision + recall != 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0
    
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }

if __name__ == "__main__":
    data_path = "data/task2_output/task2_train_bm25_2024.json"
    data = support_func.read_json(data_path)
    output = []
    for threshold in range(1, 100):
        threshold /= 100
        result = evaluate(data, threshold)
        output.append(result)
        
    df = pd.DataFrame(output)
    df.to_csv("data/task2_output/task2_train_bm25_2024.csv", index=False)

