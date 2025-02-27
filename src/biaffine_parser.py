import torch
from torch import nn

import src.config as cf
from src.modules import RecurrentEncoder, TransformerEncoder


class biaffine_parser(nn.Module):
    def __init__(self,
                 encoder, embeddings,
                 num_deprels,
                 d_arc, d_rel,
                 dropout_rate=0.33):
        """
        :param num_deprels: number of labels to be predicted
        :param d_arc: head/dependent vector states' dimension
        :param d_rel: label vector states' dimension
        :param dropout_rate: dropout rate
        """
        super(biaffine_parser, self).__init__()
        self.encoder = encoder
        self.embeddings = nn.ModuleDict(embeddings)
        self.dropout_rate = dropout_rate

        if isinstance(self.encoder, RecurrentEncoder):
            self.d_h = self.encoder.hidden_size * 2 if self.encoder.bidirectional else 1
        elif isinstance(self.encoder, TransformerEncoder):
            self.d_h = self.encoder.hidden_size + sum(emb.embedding.embedding_dim for emb in self.embeddings.values()) \
                if bool(self.embeddings) else self.encoder.hidden_size

        # Arc classifier
        self.arc_head_mlp = nn.Sequential(nn.Linear(self.d_h, d_arc), nn.ReLU(), nn.Dropout(self.dropout_rate))
        self.arc_dep_mlp = nn.Sequential(nn.Linear(self.d_h, d_arc), nn.ReLU(), nn.Dropout(self.dropout_rate))
        self.W_arc = nn.Linear(d_arc, d_arc, bias=False)
        self.bias_arc = nn.Parameter(torch.Tensor(d_arc))

        # Label classifier
        self.rel_head_mlp = nn.Sequential(nn.Linear(self.d_h, d_rel), nn.ReLU(), nn.Dropout(self.dropout_rate))
        self.rel_dep_mlp = nn.Sequential(nn.Linear(self.d_h, d_rel), nn.ReLU(), nn.Dropout(self.dropout_rate))

        self.U_rel = nn.Parameter(torch.randn(num_deprels, d_rel, d_rel))
        self.W_rel = nn.Linear(2 * d_rel, num_deprels, bias=False)
        self.bias_rel = nn.Parameter(torch.Tensor(num_deprels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_arc.weight)
        nn.init.zeros_(self.bias_arc)

        nn.init.xavier_uniform_(self.U_rel)
        nn.init.xavier_uniform_(self.W_rel.weight)
        nn.init.zeros_(self.bias_rel)

    def forward(self, training=False, *inputs):
        H = self.encoder(training, *inputs)
        H = H.float()

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
        # (B, L, d_rel) (num_deprels, d_rel, d_rel) (B, L, d_rel)^T --> (B, L, L, num_deprels)
        B, L, d = H_rel_head.shape

        S_rel_bilinear_term = torch.einsum('bld, cde, bme -> blmc', H_rel_head, self.U_rel,
                                           H_rel_dep)  # (B, L, L, num_deprels)

        # Affine term: W_rel(H_rel_head ⊕ H_rel_dep) + b
        # (2 * d_rel, num_deprels) ((B, L, d_rel)⊕(B, L, d_rel)) + (num_deprels) --> (B, L, L, num_deprels)
        H_rel_head_exp = H_rel_head.unsqueeze(2).expand(B, L, L, d)
        H_rel_dep_exp = H_rel_dep.unsqueeze(2).expand(B, L, L, d)

        concat_rel = torch.cat([H_rel_head_exp, H_rel_dep_exp], dim=-1)  # (B, L, L, 2 * d_rel)
        S_rel_affine_term = self.W_rel(concat_rel)  # (B, L, L, c)

        # Summing up the terms and Adding bias
        S_rel = S_rel_bilinear_term + S_rel_affine_term + self.bias_rel

        return S_arc, S_rel
