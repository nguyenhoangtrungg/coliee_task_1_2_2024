from core.utilities import support_func
from core.model import metrics
from core.bm25.run_bm25 import run_create_csv_bm25
from core.model import infer_model
from core.eval.eval import csv_evaluate

import torch.nn as nn
import pandas as pd

from constant import *

valid_df = run_create_csv_bm25(TESTING_PATH, LABEL_PATH, CSV_TESTING_DATA_PATH, "test", NEGATIVE_MODE, NEGATIVE_NUM)

infer_df = infer_model.encode_csv(valid_df, 1)

output_list = []

for threshold in range(1, 100):
    threshold /= 100
    result = csv_evaluate(infer_df, "bert_scoe", threshold)
    output_list.append(result)

df = pd.DataFrame(output_list)
df.to_csv("infer.csv", index=False)