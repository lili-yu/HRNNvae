from __future__ import division

import os
import argparse
import torch
import torch.nn as nn
from torch import cuda
import pandas as pd
import data_util


def main():

    #data = {'context':conversation, 'replies':replies, 'speaker':all_speaker, 'conv_turns':all_turn}

    debug = False

    smallset = False

    print('\nLoading data')
    trainfile='/D/home/lili/mnt/DATA/convaws/convdata/conv-train_v.pt' 
    if smallset:
        print('using small data set')
        trainfile='/D/home/lili/mnt/DATA/convaws/convdata/conv-test_v.pt'
    if debug:
        print('debuggggging')
        trainfile='/D/home/lili/mnt/DATA/convaws/convdata/conv-train_v_debug.pt'
    
    train_srs , train_tgt = data_util.sort_file(trainfile)
    torch.save({'context':train_srs, 'replies':train_tgt}, 'conv-train_sorted.pt')


    valfile='/D/home/lili/mnt/DATA/convaws/convdata/conv-val_v.pt'
    if debug:
        valfile='/D/home/lili/mnt/DATA/convaws/convdata/conv-val_v_debug.pt'

    val_srs , val_tgt = data_util.sort_file(valfile)
    torch.save({'context':val_srs, 'replies':val_tgt}, 'conv-val_sorted.pt')


if __name__ == "__main__":
    main()
