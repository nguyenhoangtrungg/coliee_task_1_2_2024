import torch
from transformers import AutoTokenizer

from constant import *

INPUT_MAX_LEN = 512
OUTPUT_MAX_LEN = 10

class T5Dataset:
    def __init__(self, fragment, content, label):   
        self.fragment = fragment
        self.content = content
        self.label = label
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, model_max_length=INPUT_MAX_LEN)
        self.input_max_len = INPUT_MAX_LEN
        self.output_max_len = OUTPUT_MAX_LEN
        self.promt = "Recognizing entailment between a decision fragment and a relevant legal paragraph. "

    def __len__(self):                      # This method retrives the number of item from the dataset
        return len(self.label)
    
    def tokenize_text(self, text):
        return self.tokenizer(text, max_length = self.output_max_len, padding='max_length', truncation=True)    
    def tokenize_pair_text(self, text_1, text_2):
        return self.tokenizer(text_1, text_2, max_length = self.input_max_len, padding='max_length', truncation=True)

    def __getitem__(self, item):             # This method retrieves the item at the specified index item. 

        fragment = self.promt + self.fragment[item]
        content = self.content[item]
        label = self.label[item]

        input_tokenizer = self.tokenize_pair_text(fragment, content)
        output_tokenizer = self.tokenize_text(label)
        item = {}
        item["input_ids"] = input_tokenizer["input_ids"]
        item["attention_mask"] = input_tokenizer["attention_mask"]
#         item["decoder_input_ids"] = output_tokenizer["input_ids"]
#         item["decoder_attention_mask"] = output_tokenizer["attention_mask"]
        item["labels"] = output_tokenizer["input_ids"]
        return item      