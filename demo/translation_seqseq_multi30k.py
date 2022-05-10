import argparse
import os

import torch
from torch import optim
from util import get_arch_net

from core import lossext
from core import translation


def args(training=True):
    parser = argparse.ArgumentParser(description="train and save your model")

    parser.add_argument('--arch', default='translation_seqseq', help='architecture')
    parser.add_argument('--embed-size', default=64, help='embed size')
    parser.add_argument('--enc-hidden-size', default=32, help='enc hidden size')
    parser.add_argument('--dec-hidden-size', default=32, help='dec hidden size')
    parser.add_argument('--num-layers', default=2, help='num of rnn layers')
    parser.add_argument('--src-vocab-pth', default='src_vocab.bin', help='path for the source vocab')
    parser.add_argument('--tgt-vocab-pth', default='tgt_vocab.bin', help='path for the target vocab')
    parser.add_argument('--bidirectional', default=True, help='bidirectional')
    parser.add_argument('--use-pack-padded-sequence', default=True, help='use pack padded sequence')
    parser.add_argument('--fc-dec-embed-included', default=True, help='fc decoder embed included')
    parser.add_argument('--fc-enc-context-included', default=True, help='fc encoder context included')

    if training:
        parser.add_argument('--dropout', default=0.1, help='dropout')
        parser.add_argument('--batch-size', default=64, help='batch size')
        parser.add_argument('--learning-rate', default=0.005, help='learning rate')
        parser.add_argument('--num-epochs', default=20, help="number of epochs")
        parser.add_argument('--saved-pth', default='translation_seqseq.pth', help='path for the trained model')
    else:
        parser.add_argument('--weights-pth', default='translation_seqseq.pth', help='net weights')
        parser.add_argument('--src-sentence',
                            default='Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.',
                            help='source sentence test')
        parser.add_argument('--max-num-steps', default=20, help='max no. of steps')

    return parser


def train(args: argparse.Namespace):
    root = os.path.abspath(os.path.join(os.getcwd(), '../'))  # get project root

    # nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    nw = 0
    vocabs, loaders = translation.utils.dataset.load_data_multi30k(args.batch_size, num_workers=nw,
                                                                   test_loader_included=False)
    src_vocab, tgt_vocab = vocabs
    train_loader, valid_loader, test_loader = loaders

    model_root, net = get_arch_net(
        root,
        args.arch,
        training=True,
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_size=args.embed_size,
        enc_hidden_size=args.enc_hidden_size,
        dec_hidden_size=args.dec_hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        use_pack_padded_sequence=args.use_pack_padded_sequence,
        fc_dec_embed_included=args.fc_dec_embed_included,
        fc_enc_context_included=args.fc_enc_context_included
    )

    saved_path = os.path.join(model_root, args.saved_pth)
    src_vocab_path = os.path.join(model_root, args.src_vocab_pth)
    tgt_vocab_path = os.path.join(model_root, args.tgt_vocab_pth)

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=0.001)
    loss = lossext.MaskedSoftmaxCELoss()

    translation.train_wrapper(
        net,
        train_loader,
        valid_loader,
        loss,
        optimizer,
        args.num_epochs,
        vocabs=(src_vocab, tgt_vocab),
        vocab_paths=(src_vocab_path, tgt_vocab_path),
        saved_path=saved_path)

    from matplotlib import pyplot as plt
    plt.show()


def evaluate(args: argparse.Namespace):
    root = os.path.abspath(os.path.join(os.getcwd(), '../'))  # get project root

    model_root, net, vocabs = get_arch_net(
        root,
        args.arch,
        training=False,
        vocab_pth=(args.src_vocab_pth, args.tgt_vocab_pth),
        embed_size=args.embed_size,
        enc_hidden_size=args.enc_hidden_size,
        dec_hidden_size=args.dec_hidden_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        use_pack_padded_sequence=args.use_pack_padded_sequence,
        fc_dec_embed_included=args.fc_dec_embed_included,
        fc_enc_context_included=args.fc_enc_context_included)

    weights_pth = os.path.join(model_root, args.weights_pth)
    assert os.path.exists(weights_pth), f'file {args.weights_pth} does not exist.'
    net.load_state_dict(torch.load(weights_pth, map_location='cpu'))
    # missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path, map_location='cpu'), strict=False)
    # print('[missing_keys]:', *missing_keys, sep='\n')
    # print('[unexpected_keys]:', *unexpected_keys, sep='\n')

    test_data = translation.utils.dataset.get_test_data_multi30k()

    trgs = []
    pred_trgs = []
    for i, (src, trg) in enumerate(test_data):
        pred_trg, attention_weight_seq = translation.predict_wrapper(
            net, src, vocabs[0], vocabs[1],
            max_num_steps=args.max_num_steps,
            tokenizer=translation.utils.dataset.tokenize_de
        )
        trgs.append(trg.lower().split(' '))
        pred_trgs.append(pred_trg.split(' '))

        if i >= 100:
            break

        print(f'[source]{src}')
        print(f'[target]{trg}')
        print(f'[translate]{pred_trg}')
        print(f'bleu {translation.utils.metrics.simple_bleu(pred_trg, trg, k=2):.3f}\n')

    print(f'total bleu score: {translation.utils.metrics.blue_score(pred_trgs, trgs):.3f}')


def predict(args: argparse.Namespace):
    root = os.path.abspath(os.path.join(os.getcwd(), '../'))  # get project root

    model_root, net, vocab = get_arch_net(
        root,
        args.arch,
        training=False,
        vocab_pth=(args.src_vocab_pth, args.tgt_vocab_pth),
        embed_size=args.embed_size,
        enc_hidden_size=args.enc_hidden_size,
        dec_hidden_size=args.dec_hidden_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        use_pack_padded_sequence=args.use_pack_padded_sequence,
        fc_dec_embed_included=args.fc_dec_embed_included,
        fc_enc_context_included=args.fc_enc_context_included)

    weights_pth = os.path.join(model_root, args.weights_pth)
    assert os.path.exists(weights_pth), f'file {args.weights_pth} does not exist.'
    net.load_state_dict(torch.load(weights_pth, map_location='cpu'))
    # missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path, map_location='cpu'), strict=False)
    # print('[missing_keys]:', *missing_keys, sep='\n')
    # print('[unexpected_keys]:', *unexpected_keys, sep='\n')

    ders = [
        'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.',
        'Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem.',
        'Ein Mann in einem blauen Hemd steht auf einer Leiter und putzt ein Fenster.',
        'Ein Mann lächelt einen ausgestopften Löwen an.',
        'Ein schickes Mädchen spricht mit dem Handy während sie langsam die Straße entlangschwebt.'
    ]

    for der in ders:
        translation_eng, attention_weight_seq = translation.predict_wrapper(
            net, der, vocab[0], vocab[1],
            max_num_steps=args.max_num_steps,
            tokenizer=translation.utils.dataset.tokenize_de
        )
        print(translation_eng)


if __name__ == '__main__':
    # parser = args()
    # args, unknown = parser.parse_known_args()
    # train(args)

    parser = args(training=False)
    args, unknown = parser.parse_known_args()
    evaluate(args)
    # predict(args)
