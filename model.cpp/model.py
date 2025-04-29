"""@package model
@brief This module implements neural network components.

@details This module contains classes and functions for building a neural network,
including input embedding, positional encoding, multi-head attention, and feed-forward layers.
"""

import math
import torch
import torch.nn as nn


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

    def forward(self, x: torch.Tensor):
        """@brief Forward pass for the input embedding layer.

        @param x The input tensor containing token indices of shape [batch_size, seq_len].
        @return The embedded representation of the input tokens of
        shape [batch_size, seq_len, d_model].
        """
        return self.embeddings(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """@brief Implements the Positional Encoding for the model.

    @details This class adds Input Embedding with Positional Encoding.
    Sine and Cosine functions are used here to calculate positional encoding.
    """

    def __init__(self, seq_len, d_model, dropout):
        """@brief Initializes the PositionalEncoding class.

        @param seq_len The maximum input sequence length.
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

    def forward(self, x: torch.Tensor):
        """@brief Forward pass for the positional encoding layer.

        @param x The input tensor of shape [batch_size, seq_len, d_model].
        @return Output tensor with positional encoding added with
        input embedding of shape [batch_size, seq_len, d_model].
        """
        seq_len = x.shape[1]
        x = x + (self.pos_encoding[:, :seq_len, :]).requires_grad_(False)
        return self.dropout(x)


class LayerNorm(nn.Module):
    """@brief Implements the Layer Normalization.

    @details This class normalizes MHA and Feed Forward Block.
    """

    def __init__(self, eps, d_model):
        """@brief Initializes the LayerNorm class.

        @param eps Used to avoid Divide by Zero Error
        @param d_model The dimensionality of the embeddings.
        """
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor):
        """@brief Forward pass for the Layer Normalization.

        @param x The input tensor of shape [batch_size, seq_len, d_model].
        @return Output tensor with normalized shape of [batch_size, seq_len, d_model].
        """
        mean = torch.mean(x, -1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True)

        return self.gamma * (x - mean) / (torch.sqrt(var) + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """@brief Implements the Position-wise Feed-Forward Network.

    @details This class consists of two linear transformations with
    a ReLU activation in between and dropout.
    """

    def __init__(self, d_model, d_ff, dropout):
        """@brief Initializes the FeedForwardBlock class.

        @param d_model The dimensionality of input(in_features).
        @param d_ff The inner-layer dimensionality(out_features).
        @param dropout The Probability of an element to be zeroed.
        """
        super().__init__()
        self.layer_1 = nn.Linear(d_model, d_ff, bias=True)
        self.layer_2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """@brief Forward pass for the position-wise Feed-Forward networks.

        @param x The input tensor of shape [batch_size, seq_len, d_model].
        @return Output tensor of shape [batch_size, seq_len, d_model].
        """
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_2(x)

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

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        """@brief Forward pass for Multi-Head Attention.

        @param q The query tensor of shape [batch_size, seq_len, d_model].
        @param k The key tensor of shape [batch_size, seq_len, d_model].
        @param v The value tensor of shape [batch_size, seq_len, d_model].
        @param mask Optional attention mask. Set default to None. The mask tensor of shape [batch_size, 1, 1, seq_len].
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

        x, attn_scores = MultiHeadAttention.scaled_dot_product_attention(
            query, key, value, mask, self.dropout
        )

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(x.shape[0], -1, self.num_heads * self.d_k)
        )

        return self.w_o(x)

    @staticmethod
    def scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        dropout: nn.Dropout,
    ):
        """@brief Computes scaled dot-product attention.

        @details This function calculates the attention weights based on the query,
        key, and value matrices. Optionally, a mask can be applied to ignore certain positions.

        @param query The query matrix of shape [batch_size, num_heads, seq_len, d_k].
        @param key The key matrix of shape [batch_size, num_heads, seq_len, d_k].
        @param value The value matrix of shape [batch_size, num_heads, seq_len, d_k].
        @param mask Optional mask to apply (default is None). The mask tensor of shape [batch_size, 1, 1, seq_len].
        @param dropout Optional mask to apply (default is None).
        @return Output tensor Attention output and attention scores
        of shape [batch_size, num_heads, seq_len, d_k] and
        [batch_size, num_heads, seq_len, seq_len] respectively.
        """
        d_k = query.shape[-1]

        attn_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_scores = attn_scores.softmax(dim=-1)

        if dropout is not None:
            attn_scores = dropout(attn_scores)

        return (attn_scores @ value), attn_scores


class ResidualConnection(nn.Module):
    """@brief Implements the Residual Connection Class.

    @details This class initializes LayerNorm and nn.Dropout.
    """

    def __init__(self, eps, d_model, dropout):
        """@brief Initializes the Residual Connection class.

        @param eps Epsilon for layer norm.
        @param d_model The dimensionality of the embeddings.
        @param dropout The Probability of an element to be zeroed.
        """
        super().__init__()
        self.norm = LayerNorm(eps, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """@brief Forward pass for ResidualConnection.

        @param x The input tensor of shape [batch_size, seq_len, d_model].
        @param sublayer The function implemented by the sub-layer itself which includes MHA and FFN.
        @return Output tensor of shape [batch_size, seq_len, d_model].
        """
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    """@brief Implements the EncoderBlock class.

    @details Initializes MHA -> Residual_MHA -> FFN -> Residual_FFN.
    """

    def __init__(self, d_model, num_heads, dropout, d_ff, eps):
        """@brief Initializes the EncoderBlock class.

        @param d_model The dimensionality of the embeddings.
        @param num_heads The number of heads.
        @param dropout The Probability of an element to be zeroed.
        @param d_ff The outer layer dimension.
        @param eps Epsilon for layer norm.
        """
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.residual_mha = ResidualConnection(eps, d_model, dropout)
        self.ffn = FeedForwardBlock(d_model, d_ff, dropout)
        self.residual_ffn = ResidualConnection(eps, d_model, dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """@brief Forward pass for EncoderBlock.

        @param x The input tensor of shape [batch_size, seq_len, d_model].
        @param src_mask The src_mask tensor of shape [batch_size, 1, 1, seq_len].
        @return The output tensor of shape [batch_size, seq_len, d_model].
        """
        x = self.residual_mha(x, lambda x: self.mha(x, x, x, src_mask))
        x = self.residual_ffn(x, lambda x: self.ffn(x))
        return x


class Encoder(nn.Module):
    """@brief Implements the Encoder class.

    @details Initializes InputEmbedding, PositionalEncoding and EncoderBlock.
    """

    def __init__(
        self,
        d_model,
        vocab_size,
        num_heads,
        dropout,
        d_ff,
        eps,
        num_layers,
        max_seq_len=512,
    ):
        """@brief Initializes the Encoder class.

        @param d_model The dimensionality of the embeddings.
        @param vocab_size The size of the input vocabulary.
        @param num_heads The number of parallel attention layers or heads.
        @param dropout The Probability of an element to be zeroed.
        @param d_ff The dimensionality of the inner-layer.
        @param eps Epsilon for layer norm.
        @param num_layers The number of encoder layers.
        @param max_seq_len The maximum size of input tokens.
        """
        super().__init__()
        self.input_embedding = InputEmbedding(d_model, vocab_size)
        self.positional_encoding = PositionalEncoding(max_seq_len, d_model, dropout)
        self.layers = nn.ModuleList(
            [
                EncoderBlock(d_model, num_heads, dropout, d_ff, eps)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """@brief Forward pass for Encoder class.

        @param x The input tensor of shape [batch_size, seq_len].
        @param src_mask The src_mask tensor of shape [batch_size, 1, 1, seq_len].
        @return The output tensor of shape [batch_size, seq_len, d_model].
        """
        x = self.input_embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class DecoderBlock(nn.Module):
    """@brief Implements the DecoderBlock class.

    @details Initializes Masked MHA -> Residual_Masked_MHA -> MHA -> Residual_MHA -> FFN -> Residual_FFN.
    """

    def __init__(self, d_model, num_heads, d_ff, eps, dropout):
        """@brief Initializes the DecoderBlock class.

        @param d_model The dimensionality of the embeddings.
        @param num_heads The number of heads.
        @param d_ff The outer layer dimension.
        @param eps Epsilon for layer norm.
        @param dropout The Probability of an element to be zeroed.
        """

        super().__init__()
        self.masked_mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.residual_masked_mha = ResidualConnection(eps, d_model, dropout)
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.residual_mha = ResidualConnection(eps, d_model, dropout)
        self.ffn = FeedForwardBlock(d_model, d_ff, dropout)
        self.residual_ffn = ResidualConnection(eps, d_model, dropout)

    def forward(self, x, encoder_output, tgt_mask):
        """@brief Forward pass for DecoderBlock.

        @param x The input tensor of shape [batch_size, seq_len, d_model].
        @param encoder_output The output tensor of encoder block of shape [batch_size, seq_len, d_model].
        @param tgt_mask The tgt_mask tensor of shape [1, 1, seq_len, seq_len].
        @return The output tensor of shape [batch_size, seq_len, d_model].
        """
        x = self.residual_masked_mha(
            x, lambda x: self.masked_mha(x, x, x, mask=tgt_mask)
        )
        x = self.residual_mha(x, lambda x: self.mha(x, encoder_output, encoder_output))
        x = self.residual_ffn(x, lambda x: self.ffn(x))
        return x


class Decoder(nn.Module):
    """@brief Implements the Decoder class.

    @details Initializes InputEmbedding, PositionalEncoding and DecoderBlock.
    """

    def __init__(
        self,
        d_model,
        vocab_size,
        num_heads,
        dropout,
        d_ff,
        eps,
        num_layers,
        max_seq_len=512,
    ):
        """@brief Initializes the Decoder class.

        @param d_model The dimensionality of the embeddings.
        @param vocab_size The size of the input vocabulary.
        @param num_heads The number of parallel attention layers or heads.
        @param dropout The Probability of an element to be zeroed.
        @param d_ff The dimensionality of the inner-layer.
        @param eps Epsilon for layer norm.
        @param num_layers The number of decoder layers.
        @param max_seq_len The maximum size of input tokens.
        """
        super().__init__()
        self.input_embedding = InputEmbedding(d_model, vocab_size)
        self.positional_encoding = PositionalEncoding(max_seq_len, d_model, dropout)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(d_model, num_heads, d_ff, eps, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, encoder_output, tgt_mask):
        """@brief Forward pass for Decoder class.

        @param x The input tensor of shape [batch_size, seq_len].
        @param encoder_output The Output of Encoder of shape [batch_size, seq_len , d_model].
        @param tgt_mask The tgt_mask tensor of shape [1, 1, seq_len, seq_len].
        @return The output tensor of shape [batch_size, seq_len, d_model].
        """
        x = self.input_embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask)
        return x


class Transformer(nn.Module):
    """@brief Implements the Transformer class.

    @details Initializes Encoder, Decoder class and output layer connected to decoder.
    """

    def __init__(
        self,
        d_model,
        vocab_size,
        num_heads,
        dropout,
        d_ff,
        eps,
        num_layers,
        max_seq_len,
    ):
        """@brief Initializes the Transformer class.

        @param d_model The dimensionality of the embeddings.
        @param vocab_size The size of the input vocabulary.
        @param num_heads The number of parallel attention layers or heads.
        @param dropout The Probability of an element to be zeroed.
        @param d_ff The dimensionality of the inner-layer.
        @param eps Epsilon for layer norm.
        @param num_layers The number of encoder and decoder layers.
        @param max_seq_len The maximum size of input tokens.
        """
        super().__init__()
        self.encoder = Encoder(
            d_model, vocab_size, num_heads, dropout, d_ff, eps, num_layers, max_seq_len
        )
        self.decoder = Decoder(
            d_model, vocab_size, num_heads, dropout, d_ff, eps, num_layers, max_seq_len
        )
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """@brief Forward pass for Transformer class.

        @param src The input tensor of shape [batch_size, seq_len] for encoder.
        @param tgt The input tensor of shape [batch_size, seq_len] for decoder.
        @param src_mask The src_mask tensor of shape [1, 1, seq_len, seq_len] for encoder.
        @param tgt_mask The tgt_mask tensor of shape [1, 1, seq_len, seq_len] for decoder.
        @return The output tensor of shape [batch_size, seq_len, vocab_size].
        """
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask)
        output = self.output_layer(dec_output)
        return output


def main():
    d_model = 512
    vocab_size = 10000
    seq_len = 128
    batch_size = 5
    dropout = 0.1
    num_heads = 8
    d_ff = 2048
    eps = 1e-5
    num_layers = 6
    PAD_IDX = 0

    x = torch.randint(1, 10, (batch_size, seq_len))
    src_mask = x == PAD_IDX
    src_mask = src_mask.unsqueeze(1).unsqueeze(2)
    print(f"X.shape: {x.shape}, dtype: {x.dtype}")
    print(f"src_mask.shape: {src_mask.shape}, dtype: {src_mask.dtype}")

    y = torch.randint(1, 10, (batch_size, seq_len))
    # Shift outputs to right
    y = y[:, :-1]
    new_seq_len = y.size(1)
    tgt_mask = (
        torch.tril(torch.ones(new_seq_len, new_seq_len)).unsqueeze(0).unsqueeze(1)
    )
    print(f"Y.shape: {y.shape}, dtype: {y.dtype}")
    print(f"tgt_mask.shape: {tgt_mask.shape}, dtype: {tgt_mask.dtype}")

    transformer = Transformer(
        d_model, vocab_size, num_heads, dropout, d_ff, eps, num_layers, seq_len
    )

    # Forward pass
    output = transformer(x, y, src_mask, tgt_mask)
    print(f"Transformer.shape: {output.shape}, dtype: {output.dtype}")

    total_params = sum(p.numel() for p in transformer.parameters())
    print(f"Total number of parameters = {total_params:,}")

    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"The size of the model: {total_size_mb:.2f} MB")

    for name, param in transformer.named_parameters():
        print(f"{name:60} {param.numel():,} params")


if __name__ == "__main__":
    main()
