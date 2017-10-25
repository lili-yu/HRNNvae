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
import ujson

print('starting')


def main():
    global opt
    checkpoint = None
    dict_checkpoint = opt.train_from

    trainfile='/D/home/lili/mnt/DATA/convaws/convdata/conv-train_v.json'
    train = pd.read_json(trainfile)
    print('Read training data from: {}'.format(trainfile))


    valfile='/D/home/lili/mnt/DATA/convaws/convdata/conv-val_v.json'
    val = pd.read_json(valfile)
    print('Read validation data from: {}'.format(valfile))

    train_srs = train.context.values.tolist()
    train_tgt = train.replies.values.tolist()
    val_srs = val.context.values.tolist()
    val_tgt = val.replies.values.tolist()
    
    src_vocab = hierdata.buildvocab(train_srs+val_srs)
    tgt_vocab = hierdata.buildvocab(train_tgt+val_tgt)

    with open('src_vocab.json', 'w') as outfile:
        ujson.dumps(src_vocab, outfile)
    with open('tgt_vocab.json', 'w') as outfile:
        ujson.dumps(tgt_vocab, outfile)
    

if __name__ == "__main__":
    main()
