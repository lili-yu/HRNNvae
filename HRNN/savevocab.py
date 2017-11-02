from __future__ import division

import os
import argparse
import torch
import torch.nn as nn
from torch import cuda
import pandas as pd


import data_util
import sys
#import ujson
import json

print('starting')


def main():

    debug = True
    smallset = False

    trainfile='/D/home/lili/mnt/DATA/convaws/convdata/conv-train_v.pt' 
    if debug:
        trainfile='/D/home/lili/mnt/DATA/convaws/convdata/conv-train_v_debug.pt'
    elif smallset:
        trainfile='/D/home/lili/mnt/DATA/convaws/convdata/conv-test_v.pt'
    train = torch.load(trainfile)
    print('Read training data from: {}'.format(trainfile))


    valfile='/D/home/lili/mnt/DATA/convaws/convdata/conv-val_v.pt'
    if debug:
        trainfile='/D/home/lili/mnt/DATA/convaws/convdata/conv-val_v_debug.pt'
    val = torch.load(valfile)
    print('Read validation data from: {}'.format(valfile))

    train_srs = train['context']
    train_tgt = train['replies']
    val_srs = val['context']
    val_tgt = val['replies']
    
    src_vocab, src_w = data_util.buildvocab(train_srs+val_srs)
    tgt_vocab, tgt_w = data_util.buildvocab(train_tgt+val_tgt)


    checkpoint = {'src_word2id':src_vocab, 'src_id2word':src_w, 'tgt_word2id':tgt_vocab, 'tgt_id2word':tgt_w}
    torch.save(checkpoint, 'debug_vocabs.pt')
                    

if __name__ == "__main__":
    main()
