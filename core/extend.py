# import json
# from tqdm import tqdm

# def read_json(path):
#     with open(path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     return data


# def search_in_db(query):
#     os_path_list = [
#         "/data2/cmdir/home/test01/minhnt/coliee_task_1_2_2024/cc/en/2024_coliee_en_train_NV_Embed_v1_Bi_Encoder.json",
#         "/data2/cmdir/home/test01/minhnt/coliee_task_1_2_2024/cc/en/2024_coliee_en_train_bge_en_icl_Prompt_Bi_Encoder.json",
#         "/data2/cmdir/home/test01/minhnt/coliee_task_1_2_2024/cc/en/2024_coliee_en_train_bge_m3_Bi_Encoder.json",
#         "/data2/cmdir/home/test01/minhnt/coliee_task_1_2_2024/cc/en/2024_coliee_en_train_reranker_bge_reranker_large_Cross_Encoder.json",
#         "/data2/cmdir/home/test01/minhnt/coliee_task_1_2_2024/cc/en/2024_coliee_en_train_reranker_bge_reranker_v2_m3_Cross_Encoder.json",
#         "/data2/cmdir/home/test01/minhnt/coliee_task_1_2_2024/cc/en/2024_coliee_en_train_reranker_gte_multilingual_reranker_base_Cross_Encoder.json",
#         "/data2/cmdir/home/test01/minhnt/coliee_task_1_2_2024/cc/jp/2024_coliee_jp_train_NV_Embed_v1_Bi_Encoder.json",
#         "/data2/cmdir/home/test01/minhnt/coliee_task_1_2_2024/cc/jp/2024_coliee_jp_train_bge_en_icl_Prompt_Bi_Encoder.json",
#         "/data2/cmdir/home/test01/minhnt/coliee_task_1_2_2024/cc/jp/2024_coliee_jp_train_bge_m3_Bi_Encoder.json",
#         "/data2/cmdir/home/test01/minhnt/coliee_task_1_2_2024/cc/jp/2024_coliee_jp_train_reranker_bge_reranker_large_Cross_Encoder.json",
#         "/data2/cmdir/home/test01/minhnt/coliee_task_1_2_2024/cc/jp/2024_coliee_jp_train_reranker_bge_reranker_v2_m3_Cross_Encoder.json",
#     ]
#     key_choise = [
#         "en_NV_Embed_v1_Bi_Encoder",
#         "en_icl_Prompt_Bi_Encoder",
#         "en_m3_Bi_Encoder",
#         "en_reranker_large_Cross_Encoder",
#         "en_reranker_v2_m3_Cross_Encoder",
#         "en_gte_multilingual_reranker_base_Cross_Encoder",
#         "jp_NV_Embed_v1_Bi_Encoder",
#         "jp_icl_Prompt_Bi_Encoder",
#         "jp_m3_Bi_Encoder",
#         "jp_reranker_large_Cross_Encoder",
#         "jp_reranker_v2_m3_Cross_Encoder",
#     ]
#     search_id_path = "/data2/cmdir/home/test01/minhnt/coliee_task_1_2_2024/cc/en_train_data.json"
#     search_id = read_json(search_id_path)
#     use_id = -1
#     for case in search_id:
#         if case["query"] == query:
#             use_id = case["id"]
#             break
#     if use_id == -1:
#         raise Exception("not found")
#     full_list = []
#     half_list = []
#     none_list = []
#     for i in range(len(os_path_list)):
#         os_path = os_path_list[i]
#         flag = False
#         local_case = read_json(os_path)
#         for case in local_case:
#             if case["qid"] == use_id:
#                 flag = True
#                 local_label = case["predict"]["label_id"]
#                 local_predict = case["predict"]["predict_id"][:len(local_label)]
#                 true_predict = 0
#                 for predict in local_predict:
#                     if predict in local_label:
#                         true_predict += 1
#                 if true_predict == len(local_label):
#                     full_list.append(key_choise[i])
#                 elif true_predict > 0:
#                     half_list.append(key_choise[i])
#                 else:
#                     none_list.append(key_choise[i])
#                 break
#         if flag == False:
#             print("not found", os_path)
#     try:
#         return {
#             "n_label": len(local_label),
#             "full": full_list,
#             "half": half_list,
#             "none": none_list,
#         }
#     except:
#         return {"n_label": 0, "full": full_list, "half": half_list, "none": none_list}
    

# relevant_path = "/data2/cmdir/home/test01/minhnt/coliee_task_1_2_2024/cc/2025_coliee_en_relevant.json"

# relevants = read_json(relevant_path)
# finally_predict = []
# finally_label = []

# for i_re in tqdm(range(len(relevants))):
#     qid = relevants[i_re]["qid"]
#     cnt = 0
#     local_predict_list = []
#     for j in range(len(relevants[i_re]["qids"])):
#         i_q = relevants[i_re]["qids"][j]
#         cnt += 1
#         local_query = i_q["query"]
#         local_label = i_q["labels"]
#         for k in range(len(relevants[i_re]["qids"][j]["labels"])):
#             i_l = relevants[i_re]["qids"][j]["labels"][k]
#             local_cid = i_l["cid"]
#             local_predict_list.append(local_cid)
#         relevant_query = search_in_db(local_query)
#         # relevant_query = []
#         relevants[i_re]["qids"][j]["relevant"] = relevant_query
#         if cnt == 2:
#             relevants[i_re]["qids"] = relevants[i_re]["qids"][:cnt]
#             break
#         # print(relevant_query)
#     # local_label_list = []
#     # for i_q in relevant["labels"]:
#     #     local_label_list.append(i_q["cid"])
#     # finally_predict.append(local_predict_list)
#     # finally_label.append(local_label_list)
#     # if i_re == 5:
#     #     break
#     # break

# with open("/data2/cmdir/home/test01/minhnt/coliee_task_1_2_2024/cc/hold.json", "w", encoding="utf-8") as f:
#     json.dump(relevants, f, indent=4)


import wandb
wandb.login()