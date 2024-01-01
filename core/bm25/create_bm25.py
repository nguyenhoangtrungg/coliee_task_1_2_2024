from core.utilities import support_func

import numpy as np

from rank_bm25 import BM25Okapi

def create_bm25_model(corpus):
    # content_list = create_data.get_content_list(corpus)
    content_list = corpus

    tokenized_corpus = [doc.split(" ") for doc in content_list]

    return BM25Okapi(tokenized_corpus), content_list

def single_query_bm25(query, corpus, bm25_model, topk):
    """Finding lexically relevant document in the corpus
    Using the bm25_model for retrieval

    Args:
        query (_type_): the query sequence
        corpus (_type_): the corpus, in the form of list[str]
        bm25_model (_type_): the BM25 model object
        topk (_type_): top_k for retrieval

    Returns:
        _type_: _description_
    """
    query = support_func.pre_processing(query)
    tokenized_query = query.split(" ")
    score_list = bm25_model.get_scores(tokenized_query)
    score_list = support_func.min_max_scale(score_list)
    top_list = np.argsort(score_list)[::-1][:topk]
    output_list = []
    for i in top_list:
        name_path = corpus[i].split("#$%")[0]
        local_data = {
            "content": corpus[i].split("#$%")[1],
            "name": name_path,
            "score": score_list[i],
        }
        output_list.append(local_data)
    return output_list

def multi_query_bm25(query_list, corpus, bm25_model, topk):
    output_list = []
    for query in query_list:
        print("BM25: Query " + str(query["qid"]) + " done")
        query = support_func.pre_processing(query["query_text"])
        if query == "":
            output_list.append([])
            continue
        local_output_list = single_query_bm25(query, corpus, bm25_model, topk)
        output_list.append(local_output_list)
    return output_list