import math

import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def forward(self, queries, keys, values, valid_lens=None):
        raise NotImplementedError

    def _masked_softmax(self, X, valid_lens):
        def sequence_mask(X, valid_len, value=0):
            """在序列中屏蔽不相关的项"""
            maxlen = X.size(1)
            mask = torch.arange((maxlen), dtype=torch.float32,
                                device=X.device)[None, :] < valid_len[:, None]
            X[~mask] = value
            return X

        """通过在最后一个轴上掩蔽元素来执行softmax操作"""
        # X:3D张量，valid_lens:1D或2D张量
        if valid_lens is None:
            return nn.functional.softmax(X, dim=-1)
        else:
            shape = X.shape
            if valid_lens.dim() == 1:
                valid_lens = torch.repeat_interleave(valid_lens, shape[1])
            else:
                valid_lens = valid_lens.reshape(-1)
            # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
            X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
            return nn.functional.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(Attention):

    def __init__(self, key_size, query_size, hidden_size, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)

        self.W_k = nn.Linear(key_size, hidden_size, bias=False)
        self.W_q = nn.Linear(query_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        # queries = [batch_size, no_of_queries, query_size]
        # keys = [batch_size, no_of_kv_pairs, key_size]
        # values = [batch_size, no_of_kv_pairs, value_size]
        # valid_lens = [batch_size] or [batch_size, no_of_queries]

        queries, keys = self.W_q(queries), self.W_k(keys)
        # queries = [batch_size, no_of_queries, hidden_size]
        # keys = [batch_size, no_of_kv_pairs, hidden_size]

        # broadcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        # queries => [batch_size, no_of_queries, 1, hidden_size]
        # keys => [batch_size, 1, no_of_kv_pairs, hidden_size]

        features = torch.tanh(features)
        # features = [batch_size, no_of_queries, no_of_kv_pairs, hidden_size]

        scores = self.w_v(features).squeeze(-1)
        # scores = [batch_size, no_of_queries, no_of_kv_pairs]

        self.attention_weights = self._masked_softmax(scores, valid_lens)
        # attention_weights = [batch_size, no_of_queries, no_of_kv_pairs]
        # values = [batch_size, no_of_kv_pairs, value_size]

        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(Attention):

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)

        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        # queries = [batch_size, no_of_queries, query_size]
        # keys = [batch_size, no_of_kv_pairs, key_size]
        # values = [batch_size, no_of_kv_pairs, value_size]
        # valid_lens = [batch_size] or [batch_size, no_of_queries]

        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = self._masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(Attention):

    def __init__(self, key_size, query_size, value_size, hidden_size,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)

        self.W_q = nn.Linear(query_size, hidden_size, bias=bias)
        self.W_k = nn.Linear(key_size, hidden_size, bias=bias)
        self.W_v = nn.Linear(value_size, hidden_size, bias=bias)
        self.W_o = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries = [batch_size, no_of_queries, query_size]
        # keys = [batch_size, no_of_kv_pairs, key_size]
        # values = [batch_size, no_of_kv_pairs, value_size]
        # valid_lens = [batch_size] or [batch_size, no_of_queries]

        queries = self._transpose_qkv(self.W_q(queries), self.num_heads)
        keys = self._transpose_qkv(self.W_k(keys), self.num_heads)
        values = self._transpose_qkv(self.W_v(values), self.num_heads)
        # queries = [batch_size * num_heads, no_of_queries, hidden_size/num_heads]
        # keys = [batch_size * num_heads, no_of_kv_pairs, hidden_size/num_heads]
        # values = [batch_size * num_heads, no_of_kv_pairs, hidden_size/num_heads]

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)
        # output = [batch_size * num_heads, no_of_queries, hidden_size/num_heads]

        output = self._transpose_output(output, self.num_heads)
        # output = [batch_size, no_of_queries, hidden_size]
        return self.W_o(output)

    def _transpose_qkv(self, X, num_heads):
        # X = [batch_size, no_of_queries/no_of_kv_pairs, hidden_size]

        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
        # X = [batch_size, no_of_queries/no_of_kv_pairs, num_heads, hidden_size/num_heads]

        X = X.permute(0, 2, 1, 3)
        # X = [batch_size, num_heads, no_of_queries/no_of_kv_pairs, hidden_size/num_heads]

        # outputs = [batch_size * num_heads, no_of_queries/no_of_kv_pairs, hidden_size/num_heads]
        return X.reshape(-1, X.shape[2], X.shape[3])

    def _transpose_output(self, X, num_heads):
        # X = [batch_size * num_heads, no_of_queries, hidden_size/num_heads]
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])

        X = X.permute(0, 2, 1, 3)
        # X = [batch_size, no_of_queries, num_heads, hidden_size/num_heads]

        # output = [batch_size, no_of_queries, hidden_size]
        return X.reshape(X.shape[0], X.shape[1], -1)
