"""@package model
@brief This module implements neural network components.

@details This module contains classes and functions for building a neural network,
including input embedding, positional encoding, multi-head attention, and feed-forward layers.

@todo Implement LayerNorm class. (By Ajay)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class InputEmbedding(nn.Module):
    """@brief Implements input embedding for the model.

    @details This class converts input tokens into dense vector representations
    using an embedding layer. It is typically used as the first layer in a transformer model.
    """

    def __init__(self, d_model, vocab_size):
        """@brief Initializes the InputEmbedding class.

        @param d_model The dimensionality of the embeddings.
        @param vocab_size The size of the input vocabulary.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """@brief Forward pass for the input embedding layer.

        @param x The input tensor containing token indices of shape [batch_size, seq_len].
        @return The embedded representation of the input tokens of shape [batch_size, seq_len, d_model].
        """
        return self.embeddings(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """@brief Implements the Positional Encoding for the model.

    @details This class adds Input Embedding with Positional Encoding.
    Sine and Cosine functions are used here to calculate positional encoding.
    """

    def __init__(self, seq_len, d_model, dropout):
        """@brief Initializes the PositionalEncoding class.

        @param seq_len The input tensor containing token indices.
        @param d_model The dimensionality of the embeddings.
        @param dropout The Probability of an element to be zeroed.
        """
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        pos_encoding = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0))
            / d_model
        )
        pos_encoding[:, 0::2] = torch.sin(pos * div_term)
        pos_encoding[:, 1::2] = torch.cos(pos * div_term)
        pos_encoding = pos_encoding.unsqueeze(0)

        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, x):
        """@brief Forward pass for the positional encoding layer.

        @param x The input tensor of shape [batch_size, seq_len, d_model].
        @return Output tensor with positional encoding added with input emebedding of shape [batch_size, seq_len, d_model].
        """
        seq_len = x.shape[1]
        x = x + (self.pos_encoding[:, :seq_len, :]).requires_grad_(False)
        return self.dropout(x)


class LayerNorm(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps
        self.gama = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        self.mean = torch.mean(x, -1, keepdim=True)
        self.var = torch.var(x, dim=1, keepdim=True)

        return (
            self.gama * (x - self.mean) / (torch.sqrt(self.var) + self.eps) + self.bias
        )


class FeedForwardBlock(nn.Module):
    """@brief Implements the Position-wise Feed-Forward Network.

    @details This class consists of two linear transformations with a ReLU activation in between and dropout.
    """

    def __init__(self, d_model, d_ff, dropout):
        """@brief Initializes the PositionalEncoding class.

        @param d_model The dimensionality of input(in_features).
        @param d_ff The inner-layer dimensionality(out_features).
        @param dropout The Probability of an element to be zeroed.
        """
        super().__init__()
        self.layer_1 = nn.Linear(d_model, d_ff, bias=True)
        self.layer_2 = nn.Linear(d_model, d_ff, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """@brief Forward pass for the position-wise Feed-Forward networks.

        @param x The input tensor of shape [batch_size, seq_len, d_model].
        @return Output tensor of shape [batch_size, seq_len, d_model].
        """
        x = self.layer_1(x)
        x = nn.ReLU(x)
        x = self.dropout(x)
        x = self.layer2(x)

        return x


class MultiHeadAttention(nn.Module):
    """@brief Implements the Multi-Head Attention.

    @details Projects the input into query, key, and value vectors, splits them into multiple heads,
    performs scaled dot-product attention for each head, and combines the results.
    Uses dropout for regularization and a final linear layer to project the output.
    """

    def __init__(self, d_model, num_heads, dropout):
        """@brief Initializes the MultiHeadAttention class.

        @param d_model The dimensionality of input(in_features).
        @param num_heads The inner-layer dimensionality(out_features).
        @param dropout The Probability of an element to be zeroed.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0, "d_model is not divisble by num_heads"
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)

        self.w_o = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """@brief Forward pass for Multi-Head Attention.

        @param q The query tensor of shape [batch_size, seq_len, d_model].
        @param k The key tensor of shape [batch_size, seq_len, d_model].
        @param v The value tensor of shape [batch_size, seq_len, d_model].
        @param mask Optional attention mask. Set default to None.
        @return Output tensor of shape [batch_size, seq_len, d_model].
        """
        query = self.q_proj(q)
        key = self.k_proj(k)
        value = self.v_proj(v)

        query = query.view(q.shape[0], q.shape[1], self.num_heads, self.d_k).transpose(
            1, 2
        )
        key = key.view(k.shape[0], k.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(v.shape[0], v.shape[1], self.num_heads, self.d_k).transpose(
            1, 2
        )

        x, self.attn_scores = MultiHeadAttention.scaled_dot_product_attention(
            query, key, value, mask, self.dropout
        )

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(x.shape[0], -1, self.num_heads * self.d_k)
        )

        return self.w_o(x)

    @staticmethod
    def scaled_dot_product_attention(query, key, value, mask, dropout: nn.Dropout):
        """@brief Computes scaled dot-product attention.

        @details This function calculates the attention weights based on the query,
        key, and value matrices. Optionally, a mask can be applied to ignore certain positions.

        @param query The query matrix of shape [batch_size, num_heads, seq_len, d_k].
        @param key The key matrix of shape [batch_size, num_heads, seq_len, d_k].
        @param value The value matrix of shape [batch_size, num_heads, seq_len, d_k].
        @param mask Optional mask to apply (default is None).
        @param dropout Optional mask to apply (default is None).
        @return Output tensor Attention output and attention scores
        of shape [batch_size, num_heads, seq_len, d_k] and
        [batch_size, num_heads, seq_len, seq_len] respectively.
        """
        d_k = query.shape[-1]

        attn_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attn_scores.masked_fill(mask == 0, -1e9)
        attn_scores = attn_scores.softmax(dim=-1)

        if dropout is not None:
            attn_scores = dropout(attn_scores)

        return (attn_scores @ value), attn_scores


def main():
    d_model = 50
    vocab_size = 100
    seq_len = 4
    batch_size = 1
    dropout = 0.1
    num_heads = 2
    d_k = d_model // num_heads

    x = torch.randint(1, 10, (batch_size, seq_len))
    print(f"X.shape: {x.shape}, dtype: {x.dtype}")

    ie = InputEmbedding(d_model, vocab_size)
    x = ie(x)
    print(f"InputEmbedding.shape: {x.shape}, dtype: {x.dtype}")

    pe = PositionalEncoding(seq_len, d_model, dropout)
    x = pe(x)
    print(f"PositionalEncoding.shape: {x.shape}, dtype: {x.dtype}")

    mha = MultiHeadAttention(d_model, num_heads, 0.1)
    x = mha(x, x, x)
    print(f"MultiHeadAttention.shape: {x.shape}, dtype: {x.dtype}")


if __name__ == "__main__":
    main()
