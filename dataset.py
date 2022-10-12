from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab
import torch


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        a = {}
        a['text'] = [i['text'].split() for i in samples]
        a['id'] = [i['id'] for i in samples]

        a['length'] = [len(i['text'].split()) for i in samples]
        a['text'] = self.vocab.encode_batch(a['text'], self.max_len)
        a['text'] = torch.LongTensor(a['text'])
        if 'intent' in samples[0].keys():
            a['intent'] = [self.label2idx(s['intent']) for s in samples]
            a['intent'] = torch.LongTensor(a['intent'])
        else:
            a['intent'] = ["0" for i in samples]
        # print(a['length'][0])
        # print(a['text'][0])
        # print(a['intent'][0])
        

        return a
        #raise NotImplementedError

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
