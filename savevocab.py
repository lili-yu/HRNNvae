from __future__ import division

import os
import argparse
import torch
import torch.nn as nn
from torch import cuda
import pandas as pd


import hierdata

import sys
import ujson

print('starting')


def main():


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
    
    src_vocab, src_w = hierdata.buildvocab(train_srs+val_srs)
    tgt_vocab, tgt_w = hierdata.buildvocab(train_tgt+val_tgt)

    src = {'word2id':src_vocab, 'id2word':src_w}
    tgt = {'word2id':tgt_vocab, 'id2word':tgt_w}

    with open('src.json', 'w') as outfile:
        ujson.dumps(src, outfile)
    with open('tgt.json', 'w') as outfile:
        ujson.dumps(tgt, outfile)
    

if __name__ == "__main__":
    main()
