from torch import nn


class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, enc_X, *args):
        # enc_X = [batch_size, src_len]

        # returns:
        #   outputs = [src_len, batch_size, enc_hidden_size * num_directions]
        #   state = [num_layers, batch_size, dec_hidden_size]
        raise NotImplementedError


class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, dec_X, dec_state):
        # dec_X = [batch_size, tgt_len]
        # dec_state = [num_layers, batch_size, dec_hidden_size]

        # returns:
        #   outputs = [batch_size, tgt_len, input_size]
        #   dec_state = [num_layers * num_directions, batch_size, hidden_size]
        raise NotImplementedError


class AttentionDecoder(Decoder):
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self._initialize_weights()

    def forward(self, enc_X, dec_X, *args):
        # enc_X = [batch_size, src_len]
        # dec_X = [batch_size, tgt_len]

        enc_outputs = self.encoder(enc_X, *args)
        # enc_outputs
        #   outputs = [src_len, batch_size, enc_hidden_size * num_directions]
        #   state = [num_layers, batch_size, dec_hidden_size]

        dec_state = self.decoder.init_state(enc_outputs, *args)
        # dec_state = [num_layers, batch_size, dec_hidden_size]

        # returns:
        #   outputs = [batch_size, tgt_len, input_size]
        #   dec_state = [num_layers * num_directions, batch_size, hidden_size]
        return self.decoder(dec_X, dec_state)

    def _initialize_weights(self):
        self.apply(self._xavier_init_weights)

    @staticmethod
    def _xavier_init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
        if isinstance(m, (nn.RNN, nn.GRU, nn.LSTM)):
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
