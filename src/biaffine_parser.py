import torch
from torch import nn
import src.config as cf


class biaffine_parser(nn.Module):
    def __init__(self, V_w, V_t, V_l, d_w, d_t, d_h, d_arc, d_rel, rnn_layers=3, bidirectional=True, dropout_rate=0.33):
        """
        :param V_w: word vocabulary size
        :param V_t: tag vocabulary size
        :param V_l: label vocabulary size
        :param d_w: word embedding dimension
        :param d_t: tag embedding dimension
        :param d_h: recurrent states' dimension
        :param d_arc: head/dependent vector states' dimension
        :param d_rel: label vector states' dimension
        """
        super(biaffine_parser, self).__init__()
        self.word_embeddings = nn.Embedding(V_w, d_w, padding_idx=cf.PAD_TOKEN_VAL)
        self.tag_embeddings = nn.Embedding(V_t, d_t, padding_idx=cf.PAD_TOKEN_VAL)
        self.dropout_rate = dropout_rate

        self.rnn = nn.LSTM(input_size=d_w + d_t, hidden_size=d_h, batch_first=True, bias=False,
                           bidirectional=bidirectional, dropout=self.dropout_rate, num_layers=rnn_layers)

        if bidirectional:
            d_h *= 2

        # Arc classifier
        self.arc_head_mlp = nn.Sequential(nn.Linear(d_h, d_arc), nn.ReLU(), nn.Dropout(self.dropout_rate))
        self.arc_dep_mlp = nn.Sequential(nn.Linear(d_h, d_arc), nn.ReLU(), nn.Dropout(self.dropout_rate))
        self.W_arc = nn.Linear(d_arc, d_arc, bias=False)
        self.bias_arc = nn.Parameter(torch.Tensor(d_arc))

        # Label classifier
        self.rel_head_mlp = nn.Sequential(nn.Linear(d_h, d_rel), nn.ReLU(), nn.Dropout(self.dropout_rate))
        self.rel_dep_mlp = nn.Sequential(nn.Linear(d_h, d_rel), nn.ReLU(), nn.Dropout(self.dropout_rate))

        self.U_rel = nn.Parameter(torch.randn(V_l, d_rel, d_rel))
        self.W_rel = nn.Linear(2 * d_rel, V_l, bias=False)
        self.bias_rel = nn.Parameter(torch.Tensor(V_l))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_arc.weight)
        nn.init.zeros_(self.bias_arc)

        nn.init.xavier_uniform_(self.U_rel)
        nn.init.xavier_uniform_(self.W_rel.weight)
        nn.init.zeros_(self.bias_rel)

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
        word_idx = self.dynamic_word_dropout(word_idx, training)  # Apply dynamic word dropout

        word_emb = self.word_embeddings(word_idx)
        tag_emb = self.tag_embeddings(tag_idx)

        H = torch.cat((word_emb, tag_emb), dim=-1)  # (B, L, d_t + d_w)
        H, _ = self.rnn(H)

        """
        Calculate arc score by deep bi-linear transformation as proposed in the work of "Dozat and Manning. Simpler but 
        More Accurate Semantic Dependency Parsing. Proceedings of the 56th Annual Meeting of the Association for 
        Computational Linguistics (Short Papers), pages 484–490 Melbourne, Australia, July 15 - 20, 2018.
        """

        H_arc_head = self.arc_head_mlp(H)  # (B, L, d_arc)
        H_arc_dep = self.arc_dep_mlp(H)  # (B, L, d_arc)

        W_arc_temp = self.W_arc(H_arc_dep)  # (B, L, d_arc)
        S_arc = torch.matmul(W_arc_temp, H_arc_head.transpose(1, 2))  # (B, L, L)
        S_arc += torch.matmul(H_arc_head, self.bias_arc.unsqueeze(0).T)  # Adding bias term results in final (B, L, L)

        """
        Calculate arc score by bi-affine transformation as proposed in the work of "Dozat and Manning. Simpler but More 
        Accurate Semantic Dependency Parsing. Proceedings of the 56th Annual Meeting of the Association for Computational
        Linguistics (Short Papers), pages 484–490 Melbourne, Australia, July 15 - 20, 2018.
        """
        H_rel_head = self.rel_head_mlp(H)  # (B, L, d_rel)
        H_rel_dep = self.rel_dep_mlp(H)  # (B, L, d_rel)

        # Bi-linear term: H_rel_head U H_rel_dep^T
        # (B, L, d_rel) (V_l, d_rel, d_rel) (B, L, d_rel)^T --> (B, L, L, V_l)
        B, L, d = H_rel_head.shape

        S_rel_bilinear_term = torch.einsum('bld, cde, bme -> blmc', H_rel_head, self.U_rel, H_rel_dep)  # (B, L, L, V_l)

        # Affine term: W_rel(H_rel_head ⊕ H_rel_dep) + b
        # (2 * d_rel, V_l) ((B, L, d_rel)⊕(B, L, d_rel)) + (V_l) --> (B, L, L, V_l)
        H_rel_head_exp = H_rel_head.unsqueeze(2).expand(B, L, L, d)
        H_rel_dep_exp = H_rel_dep.unsqueeze(2).expand(B, L, L, d)

        concat_rel = torch.cat([H_rel_head_exp, H_rel_dep_exp], dim=-1)  # (B, L, L, 2 * d_rel)
        S_rel_affine_term = self.W_rel(concat_rel)  # (B, L, L, c)

        # Summing up the terms and Adding bias
        S_rel = S_rel_bilinear_term + S_rel_affine_term + self.bias_rel

        return S_arc, S_rel
