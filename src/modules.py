from typing import Optional

import torch
import torch.nn as nn
import src.config as cf


class Embedding(nn.Module):
    def __init__(self, num_emb: int, emb_dim: int, padding_idx: int = cf.PAD_TOKEN_VAL):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(num_emb, emb_dim, padding_idx=padding_idx)

    def forward(self, H):
        return self.embedding(H)


class Encoder(nn.Module):
    def __init__(self, dropout=0.33):
        super(Encoder, self).__init__()
        self.dropout = dropout

    def forward(self, H):
        return H

    def dynamic_word_dropout(self, word_idx, training=False):
        """Applies word dropout before embedding lookup"""
        if training:  # Apply dropout only during training
            rand_mask = torch.rand(word_idx.shape)
            dropped_words = torch.where(rand_mask < self.dropout,
                                        torch.full_like(word_idx, cf.UNK_TOKEN_VAL),  # Replace with UNK_IDX
                                        word_idx)
            return dropped_words
        return word_idx  # No dropout during inference


class RecurrentEncoder(Encoder):
    def __init__(self, embeddings, hidden_size, rnn_type: str = "lstm", num_layers: int = 3,
                 batch_first: bool = True,
                 bidirectional: bool = True, bias: bool = False, dropout=0.33):
        super(RecurrentEncoder, self).__init__(dropout=dropout)
        self.embeddings = embeddings
        assert rnn_type in ['gru', 'lstm', 'rnn']
        self.rnn = (getattr(nn, rnn_type.upper())
                    (input_size=sum(emb.embedding.embedding_dim for emb in self.embeddings.values()),
                     hidden_size=hidden_size,
                     num_layers=num_layers,
                     batch_first=batch_first, bidirectional=bidirectional, bias=bias,
                     dropout=dropout))
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

    def forward(self, training, *inputs):
        inputs = (self.dynamic_word_dropout(inputs[0], training),) + inputs[1:]

        embedded_inputs = [embedding(input_tensor) for embedding, input_tensor in zip(self.embeddings.values(), inputs)]

        H = torch.cat(embedded_inputs, dim=-1)
        H, _ = self.rnn.forward(H)
        return H


class TransformerEncoder(Encoder):
    def __init__(self, tokenizer, pre_trained_model, embeddings, freeze: bool = True):
        super(TransformerEncoder, self).__init__()
        self.tokenizer = tokenizer
        self.transformer = pre_trained_model
        self.hidden_size = self.transformer.config.hidden_size
        self.embeddings = embeddings

        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, training, *inputs):
        inputs = (self.dynamic_word_dropout(inputs[0], training),) + inputs[1:]
        if len(self.embeddings) > 0:
            embedded_inputs = [embedding(input_tensor) for embedding, input_tensor in
                               zip(self.embeddings.values(), inputs[1:])]
            embedded_inputs = torch.cat(embedded_inputs)
            H = torch.cat([inputs[0], embedded_inputs], dim=-1)  # (B, L, d_t + d_w)
        else:
            H = torch.cat([inputs[0]], dim=-1)

        return H



