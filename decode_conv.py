from __future__ import division

import os
import argparse
import torch
import torch.nn as nn
from torch import cuda
import pandas as pd

import hiervae
import opts
import hierdata
import Loss
import Trainer
import Optim
import sys

print('starting')


parser = argparse.ArgumentParser(description='translate.py')
opts.add_md_help_argument(parser)

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-src',   required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-tgt',
                    help='True target sequence (optional)')
parser.add_argument('-output', default='pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-beam_size',  type=int, default=5,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=30,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=100,
                    help='Maximum sentence length.')
parser.add_argument('-replace_unk', action="store_true",
                    help="""Replace the generated UNK tokens with the source
                    token that had highest attention weight. If phrase_table
                    is provided, it will lookup the identified source token and
                    give the corresponding target token. If it is not provided
                    (or the identified source token does not exist in the
                    table) then it will copy the source token""")
parser.add_argument('-verbose', action="store_true",
                    help='Print scores and predictions for each sentence')
parser.add_argument('-dump_beam', type=str, default="",
                    help='File to dump beam information to.')
parser.add_argument('-n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")
parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")


def loadVOCAB():
    with open('src.json', 'r') as f:
        src = ujson.load(f)
    with open('tgt.json', 'r') as f:
        tgt = ujson.load(f)

    src_vocab = src['word2id']
    tgt_vocab = tgt['word2id']
    tgtwords = tgt['id2word']


def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object


def main():
    opt = parser.parse_args()

    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    testfile='/D/home/lili/mnt/DATA/convaws/convdata/conv-train_v.json'
    test = pd.read_json(testfile)
    print('Test training data from: {}'.format(testfile))

    test_srs = test.context.values.tolist()
    test_tgt = test.replies.values.tolist()

    with open('src.json', 'r') as f:
        src = ujson.load(f)
    with open('tgt.json', 'r') as f:
        tgt = ujson.load(f)

    src_vocab = src['word2id']
    tgt_vocab = tgt['word2id']
    tgtwords = tgt['id2word']

    config = read_config(model_config)


    test_batch_size = 16
    test_iter = hierdata.gen_minibatch(test_srs, test_tgt,  test_batch_size, src_vocab, tgt_vocab)

    checkpoint = opt.model
    print('Building model...')
    model = hiervae.make_base_model(opt, src_vocab, tgt_vocab, use_gpu(opt), checkpoint) ### Done  #### How to integrate the two embedding layers...
    print(model)
    tally_parameters(model)### Done 

    # Do training.
    train_vae(model, train_iter, valid_iter, tgt_vocab, optim)


if __name__ == "__main__":
    main()
