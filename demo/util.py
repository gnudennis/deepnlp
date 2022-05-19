import os
import pickle

from core.translation.models import seq2seq_model, seq2seq_attn_model, transformer_model


def get_arch_net(root, arch, training=True, vocab_pth=None, **kwargs):
    network_factory = {
        'translation_seqseq': seq2seq_model,
        'translation_seqseq_attn': seq2seq_attn_model,
        'translation_transformer': transformer_model,
    }
    assert arch in network_factory.keys(), f'{arch} is not supported.'

    model_root = os.path.join(root, 'saved', arch)
    os.makedirs(model_root, exist_ok=True)

    if training:
        net = network_factory[arch](**kwargs)
        print(kwargs)
        return model_root, net
    else:
        if isinstance(vocab_pth, (list, tuple)):
            # seq2seq 两个字典
            assert len(vocab_pth) == 2, 'source and target vocab path'
            vocab = [pickle.load(open(os.path.join(model_root, pth), 'rb')) for pth in vocab_pth]
            kwargs['src_vocab_size'] = len(vocab[0])
            kwargs['tgt_vocab_size'] = len(vocab[1])
        else:
            # 单一字典
            vocab = pickle.load(open(os.path.join(model_root, vocab_pth), 'rb'))
            kwargs['vocab_size'] = len(vocab)

        net = network_factory[arch](**kwargs)
        return model_root, net, vocab
