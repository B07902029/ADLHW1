from typing import Dict

import torch
from torch.nn import Embedding
from torch.nn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.embed.weight = torch.nn.Parameter(embeddings)
        self.embed_dim = embeddings.size(1)
        self.hidden_dim = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = torch.nn.LSTM(self.embed_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.LL = torch.nn.Linear(hidden_size * 2, num_class)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        # raise NotImplementedError
        # print(batch['text'])
        # print(len(batch['text']))
        vocab_embed = self.embed(batch['text'])
        #print("embedding size = ", vocab_embed.size())

        #print(batch['text'][0])

        #print()
        #print(batch['length'])
        packed_vocab_embed = pack_padded_sequence(vocab_embed, batch['length'], batch_first=True, enforce_sorted=False)
        # print("packed size = ", packed_vocab_embed.batch_sizes())

        output, _ = self.lstm(packed_vocab_embed)
        # print("lstm-output size = ", output.size())

        padded_output, padded_output_lens = pad_packed_sequence(output, batch_first=True)
        # print("padding output size = ", padded_output.size())

        last = padded_output[:, -1, :]
        # print("last =", last.size())
        output = self.LL(last)
        return output
