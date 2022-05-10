import torch

from ..utils import try_gpu

__all__ = ['predict_wrapper']

argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)


def predict_wrapper(net, src_sentence, src_vocab, tgt_vocab, max_num_steps, tokenizer,
                    device=try_gpu(), save_attention_weights=False):
    """序列到序列模型的预测"""
    special_tokens = {
        'init_token': '<sos>',
        'eos_token': '<eos>',
        'unk_token': '<unk>',
        'pad_token': '<pad>'
    }

    # 在预测时将net设置为评估模式
    net = net.to(device)
    net.eval()

    vocab_get = lambda vocab, token: vocab[token] if token in vocab else \
        vocab[special_tokens['unk_token']]
    text_transform = lambda tokenizer, vocab, sentence, lower=True: (
            [vocab[special_tokens['init_token']]] +
            [vocab_get(vocab, token) for token in tokenizer(sentence, lower)] +
            [vocab[special_tokens['eos_token']]]
    )

    src_tokens = text_transform(tokenizer, src_vocab, src_sentence, lower=True)
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)

    # 添加批量轴
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.int64, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)

    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab[special_tokens['init_token']]], dtype=torch.int64, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(max_num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab[special_tokens['eos_token']]:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.lookup_tokens(output_seq)), attention_weight_seq
