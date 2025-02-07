import os
import pickle
import sys

import torch
from torch import nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config


class biaffine_parser(nn.Module):
    def __init__(self, V_w, V_t, d_w, d_t, d_h, d):
        super(biaffine_parser, self).__init__()
        self.word_embeddings = nn.Embedding(V_w, d_w, padding_idx=config.PAD_TOKEN_VAL)
        self.tag_embeddings = nn.Embedding(V_t, d_t, padding_idx=config.PAD_TOKEN_VAL)

        self.rnn = nn.GRU(input_size=d_w + d_t, hidden_size=d_h, batch_first=True, bias=False, dropout=0.1, num_layers=2)

        self.head_mlp = nn.Sequential(nn.Linear(d_h, d), nn.ReLU(), nn.Dropout(0.1))
        self.dep_mlp = nn.Sequential(nn.Linear(d_h, d), nn.ReLU(), nn.Dropout(0.1))

        self.W_arc = nn.Linear(d, d, bias=False)

        self.bias = nn.Parameter(torch.Tensor(d))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_arc.weight)
        nn.init.zeros_(self.bias)

    def forward(self, word_idx, tag_idx):
        word_emb = self.word_embeddings(word_idx)
        tag_emb = self.tag_embeddings(tag_idx)

        H = torch.cat((word_emb, tag_emb), dim=-1)

        H, _ = self.rnn(H)

        H_arc_head = self.head_mlp(H)
        H_arc_dep = self.dep_mlp(H)
        W_temp = self.W_arc(H_arc_dep)

        S = torch.matmul(W_temp, H_arc_head.transpose(1, 2))  # (B, L, L)
        S += torch.matmul(H_arc_head, self.bias.unsqueeze(0).T)  # Adding bias term

        return S


def save_model(model : biaffine_parser, optimizer, criterion, file_path: str):
    parameters = {'model': model, 'optimizer': optimizer, 'criterion': criterion}

    with open(file_path, 'wb') as f:
        pickle.dump(parameters)

    print(f"Model saved to {file_path}")
