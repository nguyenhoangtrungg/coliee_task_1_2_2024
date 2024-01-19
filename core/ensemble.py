from core.eval.eval import csv_evaluate

def ense(infer_df, weak_list, mbert_list, t5_list):
    step = 0.01
    for w_w in range(1, 100):
        w_w = w_w / 100
        for w_m in range(1, 100):
            w_m = w_m / 100
            for w_t in range(1, 100):
                w_t = w_t / 100
                for threshold in range(70, 100):
                    threshold = threshold / 100
                    # infer_df
                    result = csv_evaluate(infer_df, "bert_scoe", threshold)
                    print(w_w, w_m, w_t, threshold)
    return 

ense([], [], []) 