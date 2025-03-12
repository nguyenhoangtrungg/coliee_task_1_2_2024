import torch
from transformers import BertTokenizer

from constant import TOKENIZER


class MultilingualBertDataset(torch.utils.data.Dataset):
   
    def __init__(self, fragment, content, labels):
        self.fragment = fragment
        self.content = content
        self.tokenizer = BertTokenizer.from_pretrained(TOKENIZER)
        try:
            if str(labels[0]) == '0' or str(labels[0]) == '1':
                self.labels = labels
            else:
                for i in range(len(labels)):
                    if str(labels[i]).lower() == 'true':
                        labels[i] = 1
                    elif str(labels[i]).lower() == 'false':
                        labels[i] = 0
                    else:
                        print("Error")
            self.labels = labels
        except:
            self.labels = labels
    def __len__(self):
        return len(self.labels)

    def tokenize_pair_text(self, text_1, text_2):
        return self.tokenizer(text_1, text_2, padding='max_length', truncation=True)

    def __getitem__(self, index):
        encodings = self.tokenize_pair_text(self.fragment[index], self.content[index])
        item = {key: torch.tensor(val) for key, val in encodings.items()}
        item['labels'] = torch.tensor(self.labels[index])
        return item