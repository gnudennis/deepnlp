import torch
from torch import nn

from .attention import AdditiveAttention
from .encoder_decoder import Encoder, Decoder, AttentionDecoder, EncoderDecoder

__all__ = ['seq2seq_model', 'seq2seq_attn_model']


class Seq2SeqEncoder(Encoder):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, num_layers,
                 dropout=0, bidirectional=True, use_pack_padded_sequence=True, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)

        self.bidirectional = bidirectional
        self.use_pack_padded_sequence = use_pack_padded_sequence
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, enc_hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1
        self.fc = nn.Linear(enc_hidden_size * num_directions, dec_hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_X, enc_valid_len=None, *args):
        # enc_X = [batch_size, src_len]

        embedded = self.dropout(self.embedding(enc_X).permute(1, 0, 2))
        # embedded = [src_len, batch_size, embed_size]

        if self.use_pack_padded_sequence:
            assert enc_valid_len is not None, 'enc_valid_lenis necessary for using pack padded sequence'

            # need to explicitly put lengths on cpu!
            packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, enc_valid_len.to('cpu'), enforce_sorted=False)

            packed_outputs, state = self.rnn(packed_embedded)
            # packed_outputs is a packed sequence containing all hidden states
            # state is now from the final non-padded element in the batch

            outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
            # outputs is now a non-packed sequence, all hidden states obtained
            #  when the input is a pad token are all zeros
        else:
            outputs, state = self.rnn(embedded)

        # outputs = [src_len, batch_size, enc_hidden_size * num_directions]
        # state   = [num_layers * num_directions, batch_size, enc_hidden_size]

        # state生成dec_state
        if self.bidirectional:
            state = state.reshape(-1, 2, state.shape[-2], state.shape[-1]).permute(0, 2, 1, 3)
            # state = [num_layers, batch_size, num_directions, enc_hidden_size]

            state = state.reshape(state.shape[0], state.shape[1], -1)
            # state = [num_layers, batch_size, num_directions * enc_hidden_size]

        state = torch.tanh(self.fc(state))
        # state = [num_layers, batch_size, dec_hidden_size]

        # outputs = [src_len, batch_size, enc_hidden_size * num_directions]
        # state = [num_layers, batch_size, dec_hidden_size]
        return outputs, state


class Seq2SeqDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, dec_hidden_size, num_layers,
                 dropout=0,
                 fc_dec_embed_included=True,
                 fc_enc_context_included=True,
                 **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)

        self.fc_dec_embed_included = fc_dec_embed_included
        self.fc_enc_context_included = fc_enc_context_included
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + dec_hidden_size, dec_hidden_size, num_layers, dropout=dropout)

        if self.fc_dec_embed_included and self.fc_enc_context_included:
            self.fc_out = nn.Linear(dec_hidden_size * 2 + embed_size, vocab_size)
        elif self.fc_dec_embed_included:
            self.fc_out = nn.Linear(dec_hidden_size + embed_size, vocab_size)
        elif self.fc_enc_context_included:
            self.fc_out = nn.Linear(dec_hidden_size * 2, vocab_size)
        else:
            self.fc_out = nn.Linear(dec_hidden_size, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def init_state(self, enc_outputs, *args):
        outputs, state = enc_outputs
        # state = [num_layers, batch_size, hidden_size]
        return state

    def forward(self, dec_X, dec_state):
        # dec_X = [batch_size, tgt_len]
        # dec_state = [num_layers, batch_size, dec_hidden_size]

        embedded = self.dropout(self.embedding(dec_X).permute(1, 0, 2))
        # embedded = [tgt_len, batch_size, embed_size]

        context = dec_state[-1].repeat(embedded.shape[0], 1, 1)
        # context = [tgt_len, batch_size, hidden_size]

        rnn_input = torch.cat((embedded, context), dim=-1)
        outputs, dec_state = self.rnn(rnn_input, dec_state)
        # outputs = [tgt_len, batch_size, hidden_size]

        if self.fc_dec_embed_included and self.fc_enc_context_included:
            outputs = torch.cat((outputs, embedded, context), dim=-1)
        elif self.fc_dec_embed_included:
            outputs = torch.cat((outputs, embedded), dim=-1)
        elif self.fc_enc_context_included:
            outputs = torch.cat((outputs, context), dim=-1)

        outputs = self.fc_out(outputs).permute(1, 0, 2)

        # outputs = [batch_size, tgt_len, input_size]
        # dec_state = [num_layers * num_directions, batch_size, hidden_size]
        return outputs, dec_state


class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, dec_hidden_size, num_layers,
                 dropout=0,
                 enc_bidirectional=True,
                 fc_dec_embed_included=False,
                 **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)

        self.fc_dec_embed_included = fc_dec_embed_included

        if enc_bidirectional:
            num_directions = 2
        else:
            num_directions = 1

        self.attention = AdditiveAttention(dec_hidden_size * num_directions, dec_hidden_size, dec_hidden_size, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + dec_hidden_size * num_directions, dec_hidden_size, num_layers, dropout=dropout)

        if self.fc_dec_embed_included:
            self.fc_out = nn.Linear(dec_hidden_size + embed_size, vocab_size)
        else:
            self.fc_out = nn.Linear(dec_hidden_size, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        outputs, hidden_state = enc_outputs

        # outputs = [src_len, batch_size, hidden_size * num_directions]
        # hidden_state = [num_layers, batch_size, hidden_size]

        outputs = outputs.permute(1, 0, 2)
        # outputs = [batch_size, src_len, hidden_size * num_directions]

        return (outputs, hidden_state, enc_valid_lens)

    def forward(self, dec_X, state):
        enc_outputs, hidden_state, enc_valid_lens = state

        # enc_outputs = [batch_size, src_len, hidden_size * num_directions]
        # hidden_state = [num_layers, batch_size, hidden_size]

        embedded = self.dropout(self.embedding(dec_X).permute(1, 0, 2))
        # embedded = [tgt_len, batch_size, embed_size]

        output, self._attention_weights = [], []

        for x in embedded:
            # x = [batch_size, embed_size]

            # 将最后一层的hidden state 作为 query
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # query = [batch_size, 1, enc_hidden_size*num_directions]

            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            # context = [batch_size, 1, enc_hidden_size*num_directions]

            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # x = [batch_size, 1, embed_size + enc_hidden_size*num_directions]

            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            # out = [1, batch_size, dec_hidden_size]
            # hidden_state = [num_layers, batch_size, dec_hidden_size]
            output.append(out)
            self._attention_weights.append(self.attention.attention_weights)

        output = torch.cat(output, dim=0)
        # output = [tgt_len, batch_size, N]

        if self.fc_dec_embed_included:
            output = torch.cat((output, embedded), dim=-1)

        output = self.fc_out(output).permute(1, 0, 2)
        # output = [batch_size, tgt_len, vocab_size]

        return output, [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights


def seq2seq_model(
        src_vocab_size,
        tgt_vocab_size,
        embed_size,
        enc_hidden_size,
        dec_hidden_size,
        num_layers,
        dropout=0,
        bidirectional=True,
        use_pack_padded_sequence=True,
        fc_dec_embed_included=True,
        fc_enc_context_included=True
) -> EncoderDecoder:
    encoder = Seq2SeqEncoder(src_vocab_size, embed_size, enc_hidden_size, dec_hidden_size, num_layers,
                             dropout=dropout,
                             bidirectional=bidirectional,
                             use_pack_padded_sequence=use_pack_padded_sequence)
    decoder = Seq2SeqDecoder(tgt_vocab_size, embed_size, dec_hidden_size, num_layers,
                             dropout=dropout,
                             fc_dec_embed_included=fc_dec_embed_included,
                             fc_enc_context_included=fc_enc_context_included)
    net = EncoderDecoder(encoder, decoder)
    return net


def seq2seq_attn_model(
        src_vocab_size,
        tgt_vocab_size,
        embed_size,
        enc_hidden_size,
        dec_hidden_size,
        num_layers,
        dropout=0,
        bidirectional=True,
        use_pack_padded_sequence=True,
        fc_dec_embed_included=True
) -> EncoderDecoder:
    encoder = Seq2SeqEncoder(src_vocab_size, embed_size, enc_hidden_size, dec_hidden_size, num_layers,
                             dropout=dropout,
                             bidirectional=bidirectional,
                             use_pack_padded_sequence=use_pack_padded_sequence)
    decoder = Seq2SeqAttentionDecoder(tgt_vocab_size, embed_size, dec_hidden_size, num_layers,
                                      dropout=dropout,
                                      enc_bidirectional=bidirectional,
                                      fc_dec_embed_included=fc_dec_embed_included)

    net = EncoderDecoder(encoder, decoder)
    return net
