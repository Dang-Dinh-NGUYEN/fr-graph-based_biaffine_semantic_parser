import torch
from torch import nn
import src.config as cf


class biaffine_parser(nn.Module):
    def __init__(self, V_w, V_t, d_w, d_t, d_h, d, dropout_rate=0.33):
        """
        :param V_w: word vocabulary size
        :param V_t: tag vocabulary size
        :param d_w: word embedding dimension
        :param d_t: tag embedding dimension
        :param d_h: hidden dimension
        :param d: MLP hidden dimension
        """
        super(biaffine_parser, self).__init__()
        self.word_embeddings = nn.Embedding(V_w, d_w, padding_idx=cf.PAD_TOKEN_VAL)
        self.tag_embeddings = nn.Embedding(V_t, d_t, padding_idx=cf.PAD_TOKEN_VAL)
        self.dropout_rate = dropout_rate

        self.rnn = nn.GRU(input_size=d_w + d_t, hidden_size=d_h, batch_first=True, bias=False, dropout=self.dropout_rate, num_layers=2)

        self.head_mlp = nn.Sequential(nn.Linear(d_h, d), nn.ReLU(), nn.Dropout(self.dropout_rate))
        self.dep_mlp = nn.Sequential(nn.Linear(d_h, d), nn.ReLU(), nn.Dropout(self.dropout_rate))

        self.W_arc = nn.Linear(d, d, bias=False)

        self.bias = nn.Parameter(torch.Tensor(d))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_arc.weight)
        nn.init.zeros_(self.bias)

    def dynamic_word_dropout(self, word_idx, training=False):
        """Applies word dropout before embedding lookup"""
        if training:  # Apply dropout only during training
            rand_mask = torch.rand(word_idx.shape)
            dropped_words = torch.where(rand_mask < self.dropout_rate,
                                        torch.full_like(word_idx, cf.UNK_TOKEN_VAL),  # Replace with UNK_IDX
                                        word_idx)
            return dropped_words
        return word_idx  # No dropout during inference

    def forward(self, word_idx, tag_idx, training=False):
        # Apply dynamic word dropout
        word_idx = self.dynamic_word_dropout(word_idx, training)

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
