from typing import Optional

import torch
import torch.nn as nn
import src.config as cf
from termcolor import colored


class Embedding(nn.Module):
    def __init__(self, num_emb: int, emb_dim: int, padding_idx: int = cf.PAD_TOKEN_VAL):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(num_emb, emb_dim, padding_idx=padding_idx)

    def forward(self, H):
        return self.embedding(H)


class Encoder(nn.Module):
    def __init__(self, dropout=0.33,  device=None):
        super(Encoder, self).__init__()
        self.pad_token = cf.PAD_TOKEN
        self.pad_token_id = cf.PAD_TOKEN_VAL
        self.unk_token = cf.UNK_TOKEN
        self.unk_token_id = cf.UNK_TOKEN_VAL

        self.dropout = dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, H):
        return H


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

    
    def dynamic_word_dropout(self, word_idx, training=False):
        """Applies word dropout before embedding lookup"""
        if training:  # Apply dropout only during training
            non_pad_mask = word_idx != cf.PAD_TOKEN_VAL  # Mask to identify non-padding tokens
            non_pad_word_idx = word_idx[non_pad_mask]  # Select only non-PAD tokens

            # Generate dropout mask only for non-PAD tokens
            rand_mask = torch.rand(non_pad_word_idx.shape, device=self.device)  

            # Apply dropout to non-PAD tokens
            dropped_non_pad_words = torch.where(rand_mask < self.dropout,
                                                torch.full_like(non_pad_word_idx, cf.UNK_TOKEN_VAL),  
                                                non_pad_word_idx)

            # Restore original tensor shape
            dropped_words = word_idx.clone()
            dropped_words[non_pad_mask] = dropped_non_pad_words

            return dropped_words

        return word_idx  # No dropout during inference


class TransformerEncoder(Encoder):
    def __init__(self, tokenizer, pre_trained_model, embeddings, unfreeze: int = 0):
        super(TransformerEncoder, self).__init__()
        self.tokenizer = tokenizer
        self.transformer = pre_trained_model
        self.hidden_size = self.transformer.config.hidden_size
        self.embeddings = embeddings

        
        for param in self.transformer.parameters():
            param.requires_grad = False  # Freeze all layers

        if unfreeze > 0:
            print(colored(f"Unfreeze last {unfreeze} layers!", "red"))
            # Unfreeze last `n` encoder layers
            for layer in self.transformer.encoder.layer[unfreeze:]:  # Last n layers
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Tokenize the UNK and PAD tokens separately
        self.pad_token = self.tokenizer.pad_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.unk_token = self.tokenizer.unk_token
        self.unk_token_id = self.tokenizer.unk_token_id

        unk_token_input = self.tokenizer(self.tokenizer.unk_token, return_tensors="pt").to(self.device)
        pad_token_input = self.tokenizer(self.tokenizer.pad_token, return_tensors="pt").to(self.device)

        # Pass through the transformer model to get contextual embeddings
        with torch.no_grad():  # No need for gradients
            unk_hidden_states = self.transformer(**unk_token_input)['last_hidden_state']
            pad_hidden_states = self.transformer(**pad_token_input)['last_hidden_state']

        # Extract embeddings corresponding to the UNK and PAD tokens
        self.unk_token_embedding = unk_hidden_states[:, 0, :] # Shape: (D,)
        self.pad_token_embedding = pad_hidden_states[:, 0, :]  # Shape: (D,)


    def forward(self, training, *inputs):
        
        inputs = inputs[:-1] + (self.dynamic_word_dropout(inputs[-1], training=training),)
       
        if len(self.embeddings) > 0:
            embedded_inputs = [embedding(input_tensor) for embedding, input_tensor in
                               zip(self.embeddings.values(), inputs[:-1])]
            embedded_inputs = torch.cat(embedded_inputs)
            H = torch.cat([inputs[-1], embedded_inputs], dim=-1)  # (B, L, d_t + d_w)
        else:
            H = torch.cat([inputs[-1]], dim=-1)

        return H


    def dynamic_word_dropout(self, word_embeddings, dropout_rate=0.2, training=True):
        
        if not training:
            return word_embeddings  # No dropout during inference

        B, L, D = word_embeddings.shape  

        # Step 1: Identify PAD tokens
        pad_mask = (word_embeddings == self.pad_token_embedding).all(dim=-1)  # Shape: (B, L), True for PAD positions

        # Step 2: Create dropout mask (B, L) → Drop only non-PAD words
        dropout_mask = (torch.rand(B, L, device=word_embeddings.device) < dropout_rate) & ~pad_mask

        # Step 3: Ensure UNK embedding has correct dtype & device
        self.unk_token_embedding = self.unk_token_embedding.to(word_embeddings.dtype)

        # Step 4: Expand UNK embedding to (B, L, D)
        unk_expanded = self.unk_token_embedding.expand(B, L, D)

        # Step 5: Replace selected words with UNK embedding using torch.where
        dropped_embeddings = torch.where(dropout_mask.unsqueeze(-1), unk_expanded, word_embeddings)

        return dropped_embeddings


