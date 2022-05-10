import math
import pickle

import torch
from torch import nn

from ..utils import try_all_gpus, Accumulator, Animator, Timer

__all__ = ['train_wrapper', 'evaluate']


def grad_clipping(net, theta):
    """Clip the gradient."""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_batch(net, batch, tgt_vocab, loss, optimizer, devices):
    enc_X, enc_valid_len, Y, dec_valid_len = [x.to(devices[0]) for x in batch]
    dec_X = Y[:, :-1]
    dec_output = Y[:, 1:]

    net.train()
    optimizer.zero_grad()

    Y_hat, _ = net(enc_X, dec_X, enc_valid_len)
    l = loss(Y_hat, dec_output, dec_valid_len)

    l.sum().backward()  # 损失函数的标量进行“反向传播”
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
    # grad_clipping(net, 1)
    optimizer.step()

    with torch.no_grad():
        train_loss_sum = l.sum()
        num_tokens = dec_valid_len.sum()
    return train_loss_sum, enc_X.shape[0], num_tokens


def evaluate(net, data_loader, loss, devices=try_all_gpus()):
    net.eval()

    metric = Accumulator(2)  # Sum of loss, no. of tokens
    for i, batch in enumerate(data_loader):
        enc_X, enc_valid_len, Y, dec_valid_len = [x.to(devices[0]) for x in batch]
        dec_X = Y[:, :-1]
        dec_output = Y[:, 1:]
        Y_hat, _ = net(enc_X, dec_X, enc_valid_len)
        l = loss(Y_hat, dec_output, dec_valid_len)
        metric.add(l.sum(), enc_X.shape[0])
    l = metric[0] / metric[1]
    ppl = math.exp(l)

    return l, ppl


def train_wrapper(
        net,
        train_loader,
        valid_loader,
        loss,
        optimizer,
        num_epochs,
        devices=try_all_gpus(),
        vocabs=None,
        vocab_paths=None,
        saved_path=None,
        test_loader=None
):
    """Train a model with mutiple GPUs"""
    # save vocab
    src_vocab, tgt_vocab = vocabs

    if vocab_paths:
        src_vocab_path, tgt_vocab_path = vocab_paths
        with open(src_vocab_path, 'wb') as f:
            pickle.dump(src_vocab, f)
        with open(tgt_vocab_path, 'wb') as f:
            pickle.dump(tgt_vocab, f)

    timer = Timer()
    # num_batches = len(train_loader)
    animator = Animator(xlabel='epoch', ylabel='perplexity', xlim=[1, num_epochs],
                        legend=['train ppl', 'valid ppl'])

    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    best_valid_loss = float('inf')

    for epoch in range(num_epochs):
        net.train()

        metric = Accumulator(3)  # Sum of training loss, batch samples, no. of tokens

        for i, batch in enumerate(train_loader):
            timer.start()
            train_loss_sum, num_examples, num_tokens = train_batch(net, batch, tgt_vocab, loss, optimizer, devices)
            metric.add(train_loss_sum, num_examples, num_tokens)
            timer.stop()

            if (i + 1) % 50 == 0:
                train_loss = metric[0] / metric[1]
                train_ppl = math.exp(train_loss)
                print(f'iter {i + 1:3} of epoch {epoch + 1:02} | '
                      f'train ppl {train_ppl:7.3f}, '
                      f'train loss {train_loss:7.3f}')
        train_loss = metric[0] / metric[1]
        train_ppl = math.exp(train_loss)
        valid_loss, valid_ppl = evaluate(net, valid_loader, loss, devices)
        animator.add(epoch + 1, (train_ppl, valid_ppl))

        best = False
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best = True
        if best:
            print(f'[best] epoch {epoch + 1:02} | '
                  f'train ppl {train_ppl:7.3f}, '
                  f'train loss {train_loss:7.3f}, ',
                  f'valid ppl {valid_ppl:.1f}, '
                  f'valid loss {valid_loss:.3f}')
            if saved_path:
                torch.save(net.module.state_dict(), saved_path)
        else:
            print(f'epoch {epoch + 1:02} | '
                  f'train ppl {train_ppl:7.3f}, '
                  f'train loss {train_loss:7.3f}, ',
                  f'valid ppl {valid_ppl:.1f}, '
                  f'valid loss {valid_loss:.3f}')

    print(f'{metric[2] * num_epochs / timer.sum():.1f} tokens/sec on {str(devices)}')
