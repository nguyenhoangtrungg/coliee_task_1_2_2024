def ense(weak_list, mbert_list, t5_list):
    step = 0.01
    for w_w in range(1, 100):
        w_w = w_w / 100
        for w_m in range(1, 100):
            w_m = w_m / 100
            for w_t in range(1, 100):
                w_t = w_t / 100
                for threshold in range(50, 100):
                    threshold = threshold / 100
                    print(w_w, w_m, w_t, threshold)
    return 

ense(1, 2, 3)