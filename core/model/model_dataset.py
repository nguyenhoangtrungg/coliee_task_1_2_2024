import torch
from transformers import BertTokenizer

from constant import PRETRAIN_MODEL, TOKENIZER


class MultilingualBertDataset(torch.utils.data.Dataset):
   
    def __init__(self, fragment, content, labels):
        self.fragment = fragment
        self.content = content
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained(TOKENIZER)

    def __len__(self):
        return len(self.labels)

    def tokenize_pair_text(self, text_1, text_2):
        return self.tokenizer(text_1, text_2, padding='max_length', truncation=True)

    def __getitem__(self, index):
        encodings = self.tokenize_pair_text(self.fragment[index], self.content[index])
        item = {key: torch.tensor(val) for key, val in encodings.items()}
        item['labels'] = torch.tensor(self.labels[index])
        return item