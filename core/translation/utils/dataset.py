import os
from collections import Counter, OrderedDict

import spacy
from torchtext.datasets import Multi30k
from torchtext.vocab import vocab

from .data import TranslationDataLoader

__all__ = ['load_data_multi30k', 'tokenize_de', 'tokenize_en', 'get_test_data_multi30k']

# python -m spacy download en_core_web_sm
# python -m spacy download zh_core_web_sm
# or https://spacy.io/models/zh#zh_core_web_sm
# pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz
# pip install https://github.com/explosion/spacy-models/releases/download/zh_core_web_sm-3.3.0/zh_core_web_sm-3.3.0.tar.gz

spacy_de = spacy.load('de_core_news_md')
spacy_en = spacy.load('en_core_web_sm')

_special_tokens = {
    'init_token': '<sos>',
    'eos_token': '<eos>',
    'unk_token': '<unk>',
    'pad_token': '<pad>'
}


def tokenize_de(text, lower=True):
    """
    Tokenizes German text from a string into a list of strings
    """
    tokens = [tok.text for tok in spacy_de.tokenizer(text)]
    if lower:
        tokens = [token.lower() for token in tokens]
    return tokens


def tokenize_en(text, lower=True):
    """
    Tokenizes English text from a string into a list of strings
    """
    tokens = [tok.text for tok in spacy_en.tokenizer(text)]
    if lower:
        tokens = [token.lower() for token in tokens]
    return tokens


def load_data_multi30k(batch_size, num_workers=0, lower=True,
                       valid_loader_included=True, test_loader_included=True):
    train_data, valid_data, test_data = Multi30k(
        root=os.path.expanduser('~/.torchtext/cache'),
        split=('train', 'valid', 'test'),
        language_pair=("de", "en")
    )

    src_counter = Counter()
    tgt_counter = Counter()
    train_size = 0
    for (src, tgt) in train_data:
        src_counter.update(tokenize_de(src, lower=True))
        tgt_counter.update(tokenize_en(tgt, lower=True))
        train_size += 1
    print(f"train size: {train_size}")

    src_counter = OrderedDict(sorted(src_counter.items(), key=lambda x: x[1], reverse=True))
    tgt_counter = OrderedDict(sorted(tgt_counter.items(), key=lambda x: x[1], reverse=True))
    src_vocab = vocab(src_counter, min_freq=2, specials=tuple(_special_tokens.values()))
    tgt_vocab = vocab(tgt_counter, min_freq=2, specials=tuple(_special_tokens.values()))
    print(f"Unique tokens in source (de) vocabulary: {len(src_vocab)}")
    print(f"Unique tokens in target (en) vocabulary: {len(tgt_vocab)}")

    train_loader = TranslationDataLoader(
        train_data,
        tokenizers=(tokenize_de, tokenize_en),
        vocabs=(src_vocab, tgt_vocab),
        batch_size=batch_size,
        special_tokens=_special_tokens,
        include_lengths=True,
        shuffle=True,
        lower=lower,
        num_workers=num_workers
    )

    loaders = [train_loader]
    if valid_loader_included:
        valid_loader = TranslationDataLoader(
            valid_data,
            tokenizers=(tokenize_de, tokenize_en),
            vocabs=(src_vocab, tgt_vocab),
            batch_size=batch_size,
            special_tokens=_special_tokens,
            include_lengths=True,
            shuffle=False,
            lower=lower,
            num_workers=num_workers
        )
        loaders.append(valid_loader)

    if test_loader_included:
        test_loader = TranslationDataLoader(
            test_data,
            tokenizers=(tokenize_de, tokenize_en),
            vocabs=(src_vocab, tgt_vocab),
            batch_size=batch_size,
            special_tokens=_special_tokens,
            include_lengths=True,
            shuffle=False,
            lower=lower,
            num_workers=num_workers
        )
        loaders.append(test_loader)

    vocabs = (src_vocab, tgt_vocab)
    loaders = tuple(loaders)
    return vocabs, loaders


def get_test_data_multi30k():
    _, _, test_data = Multi30k(
        root=os.path.expanduser('~/.torchtext/cache'),
        split=('train', 'valid', 'test'),
        language_pair=("de", "en")
    )

    return test_data
