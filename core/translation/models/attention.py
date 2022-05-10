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
    """加性注意力"""

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
    """缩放点积注意力"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)

        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = self._masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
