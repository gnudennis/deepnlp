import math

import torch
from torch import nn

from .attention import MultiHeadAttention
from .encoder_decoder import Encoder, AttentionDecoder, EncoderDecoder

__all__ = ['transformer_model']


class PositionalEncoding(nn.Module):

    def __init__(self, hidden_size, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, hidden_size))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, hidden_size, 2, dtype=torch.float32) / hidden_size)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


## test
# from core.utils.visual import plt, show_heatmaps
#
# encoding_dim, num_steps = 32, 60
# pos_encoding = PositionalEncoding(encoding_dim, 0)
# pos_encoding.eval()
# X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
# P = pos_encoding.P[:, :X.shape[1], :]
#
# P = P[0, :, :].unsqueeze(0).unsqueeze(0)
# show_heatmaps(P, xlabel='Column (encoding dimension)',
#               ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
#
# plt.show()


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_input_size, ffn_hidden_size, ffn_outputs_size,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_input_size, ffn_hidden_size)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_hidden_size, ffn_outputs_size)

    def forward(self, X):
        # X = [batch_size, len, ffn_input_size]
        # ouputs = [batch_size, len, ffn_outputs_size]
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, hidden_size,
                 norm_shape, ffn_input_size, ffn_hidden_size, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)

        self.attention = MultiHeadAttention(key_size, query_size, value_size, hidden_size, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_input_size, ffn_hidden_size, hidden_size)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, hidden_size,
                 norm_shape, ffn_input_size, ffn_hidden_size, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)

        self.i = i
        self.attention1 = MultiHeadAttention(key_size, query_size, value_size, hidden_size, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(key_size, query_size, value_size, hidden_size, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_input_size, ffn_hidden_size, hidden_size)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]

        # 训练阶段，输出序列的所有词元都在同一时间处理，state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)

        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头:(batch_size,num_steps), 其中每一行是[1,2,...,num_steps]
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # masked multi-head self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)

        # encoder-decoder multi-head attention
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)

        return self.addnorm3(Z, self.ffn(Z)), state


class TransformerEncoder(Encoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 hidden_size, norm_shape, ffn_input_size, ffn_hidden_size,
                 num_heads, num_layers,
                 dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                f'block{i}',
                EncoderBlock(key_size, query_size, value_size, hidden_size,
                             norm_shape, ffn_input_size, ffn_hidden_size,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # embedding matrix的初始化方式是xavier init，这种方式的方差是1/embedding size，
        # 因此乘以embedding size的开方使得embedding matrix的方差是1，在这个scale下可能更有利于embedding matrix的收敛。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.hidden_size))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X


class TransformerDecoder(AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 hidden_size, norm_shape, ffn_input_size, ffn_outputs_size,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                f'block{i}',
                DecoderBlock(key_size, query_size, value_size, hidden_size,
                             norm_shape, ffn_input_size, ffn_outputs_size,
                             num_heads, dropout, i))
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.hidden_size))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.fc(X), state

    @property
    def attention_weights(self):
        return self._attention_weights


def transformer_model(
        src_vocab_size,
        tgt_vocab_size,
        hidden_size,
        ffn_hidden_size,
        num_heads,
        num_layers,
        dropout=0
) -> EncoderDecoder:
    norm_shape = [hidden_size]

    encoder = TransformerEncoder(
        src_vocab_size, hidden_size, hidden_size, hidden_size,
        hidden_size, norm_shape, hidden_size, ffn_hidden_size,
        num_heads, num_layers, dropout)

    decoder = TransformerDecoder(
        tgt_vocab_size, hidden_size, hidden_size, hidden_size,
        hidden_size, norm_shape, hidden_size, ffn_hidden_size,
        num_heads, num_layers, dropout)

    net = EncoderDecoder(encoder, decoder)
    return net
