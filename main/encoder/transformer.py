import math
import os

# import lightning as L
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class SimpleAttention(nn.Module):
    def __init__(
        self,
        emb_size: int = 128,
        key_hidden_size: int = 256,
        value_hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.key_hidden_size = key_hidden_size
        self.value_hidden_size = value_hidden_size

        self.tokeys = nn.Linear(emb_size, key_hidden_size, bias=False)
        self.toqueries = nn.Linear(emb_size, key_hidden_size, bias=False)
        self.tovalues = nn.Linear(emb_size, value_hidden_size, bias=False)
        # self.tovalues = nn.Identity()
        # self.output_layer = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        :param x: Vectors that will be used as keys, values, queries.
                  [batch_size x seq_len x embedding_size]
        :param mask: Mask that will 'remove' the attention from some
                  of the key, value vectors. [batch_size x 1 x key_len]

        :return:
            - Returns a [batch x seq_len x embedding_size] with the contextualized
                representations of the queries.
        """
        b, t, e = x.size()
        h = 1
        assert (
            e == self.emb_size
        ), f"Input embedding dim ({e}) should match layer embedding dim ({self.emb_size})"

        keys = self.tokeys(x).view(b, t, h, self.key_hidden_size).transpose(1, 2)
        queries = self.toqueries(x).view(b, t, h, self.key_hidden_size).transpose(1, 2)
        values = self.tovalues(x).view(b, t, h, self.value_hidden_size).transpose(1, 2)

        # compute scaled dot-product self-attention
        queries = queries / math.sqrt(self.key_hidden_size)

        # for each word Wi the score with all other words Wj
        # for all heads inside the batch
        # [batch x num_heads x seq_len x seq_len]
        dot = torch.matmul(queries, keys.transpose(2, 3))

        # apply the mask (if we have one)
        # We add a dimension for the heads to it below: [batch, 1, 1, seq_len]
        if mask is not None:
            dot = dot.masked_fill(~mask.unsqueeze(1), float("-inf"))

        # apply attention to convert the dot scores into probabilities.
        attention = F.softmax(dot, dim=-1)

        # We multiply the probabilities with the respective values
        context = torch.matmul(attention, values)
        # Finally, we reshape back to [batch x seq_len x num_heads * embedding_size]
        context = (
            context.transpose(1, 2).contiguous().view(b, t, h * self.value_hidden_size)
        )
        # We unify the heads by appliying a linear transform from:
        # [batch x seq_len x num_heads * embedding_size] -> [batch x seq_len x embedding_size]

        return context


class ShallowTransformer(nn.Module):
    "Simple Attention Layer"

    def __init__(
        self,
        emb_size: int = 128,
        seq_length: int = 256,
        num_classes: int = 2,
        dropout=0.1,
        key_hidden_size=None,
        value_hidden_size=None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        key_hidden_size = key_hidden_size
        value_hidden_size = value_hidden_size
        if key_hidden_size is None:
            key_hidden_size = emb_size
        elif value_hidden_size is None:
            value_hidden_size = emb_size

        # self.pos_embedding = nn.Embedding(embedding_dim=emb_size, num_embeddings=seq_length)
        self.pos_embedding = PositionalEncoding(
            emb_size, dropout=dropout, max_len=seq_length
        )
        self.attention = SimpleAttention(
            emb_size=emb_size,
            key_hidden_size=key_hidden_size,
            value_hidden_size=value_hidden_size,
        )
        self.do = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = x
        # b, t, e = tokens.size()
        # positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        positions = self.pos_embedding(tokens.permute(1, 0, 2)).permute(1, 0, 2)
        x = tokens + positions
        x = self.do(x)
        x = self.attention(x, mask)
        x = self.do(x)
        x = x.mean(dim=1)
        return x

class DeepTransformer(nn.Module):
    "Deep Attention Layer"

    def __init__(
        self,
        emb_size: int = 128,
        seq_length: int = 256,
        dropout=0.1,
        key_hidden_size=None,
        value_hidden_size=None,
        num_blocks=4,  # Number of attention blocks to repeat
    ) -> None:
        super().__init__()
        key_hidden_size = key_hidden_size
        value_hidden_size = value_hidden_size
        if key_hidden_size is None:
            key_hidden_size = emb_size
        elif value_hidden_size is None:
            value_hidden_size = emb_size

        self.pos_embedding = PositionalEncoding(
            emb_size, dropout=dropout, max_len=seq_length
        )
        
        # Define attention blocks
        self.attention_blocks = nn.ModuleList([
            SimpleAttention(
                emb_size=emb_size,
                key_hidden_size=key_hidden_size,
                value_hidden_size=value_hidden_size,
            ) for _ in range(num_blocks)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = x
        positions = self.pos_embedding(tokens.permute(1, 0, 2)).permute(1, 0, 2)
        x = tokens + positions
        x = self.dropout(x)

        # Apply multiple attention blocks
        for attention_block in self.attention_blocks:
            x = attention_block(x, mask)
            x = self.dropout(x)

        x = x.mean(dim=1)
        return x


def d(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if tensor.is_cuda else "cpu"
