import torch
from torch.utils.data import DataLoader

__all__ = ['TranslationDataLoader']


class TranslationDataLoader(DataLoader):
    def __init__(self,
                 dataset,
                 tokenizers,
                 vocabs,
                 batch_size,
                 special_tokens,
                 include_lengths=True,
                 shuffle=False,
                 lower=True,
                 num_workers=0):
        assert len(tokenizers) == 2, 'must load source and target tokenizers'
        assert len(vocabs) == 2, 'must load source and target vocabularies'
        self.src_tokenizer, self.tgt_tokenizer = tokenizers
        self.src_vocab, self.tgt_vocab = vocabs

        self.special_tokens = special_tokens
        self.include_lengths = include_lengths
        self.shuffle = shuffle
        self.lower = lower

        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                         collate_fn=self._collate_fn_wrapper())

    def _collate_fn_wrapper(self):
        def _collate_fn(batch):
            assert len(batch[0]) == 2, 'each of batch must has source and target sentences'

            def build_array_nmt(tokenizer, vocab, sentences, is_target=False):
                vocab_get = lambda vocab, token: vocab[token] if token in vocab else vocab[
                    self.special_tokens['unk_token']]
                text_transform = lambda tokenizer, vocab, sentence, lower=True: (
                        [vocab[self.special_tokens['init_token']]] +
                        [vocab_get(vocab, token) for token in tokenizer(sentence, lower)] +
                        [vocab[self.special_tokens['eos_token']]]
                )
                array = [text_transform(tokenizer, vocab, sentence, self.lower) for sentence in sentences]
                valid_len = [len(sentence) for sentence in array]

                def truncate_pad(tokens, num_steps, vocab):
                    """Truncate or pad sequences."""
                    if len(tokens) > num_steps:
                        return tokens[:num_steps]  # Truncate
                    return tokens + [vocab[self.special_tokens['pad_token']]] * (num_steps - len(tokens))  # Pad

                num_steps = max(valid_len)
                array = torch.tensor([truncate_pad(tokens, num_steps, vocab) for tokens in array], dtype=torch.int64)
                if is_target:
                    valid_len = [x - 1 for x in valid_len]

                valid_len = torch.tensor(valid_len, dtype=torch.int64)
                return array, valid_len

            source = [parts[0] for parts in batch]
            target = [parts[1] for parts in batch]
            src_array, src_valid_len = build_array_nmt(self.src_tokenizer, self.src_vocab, source)
            tgt_array, tgt_valid_len = build_array_nmt(self.tgt_tokenizer, self.tgt_vocab, target, is_target=True)

            if self.include_lengths:
                data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
            else:
                data_arrays = (src_array, tgt_array)
            return data_arrays

        return _collate_fn
