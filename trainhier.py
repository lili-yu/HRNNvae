from __future__ import division

import os
import argparse
import torch
import torch.nn as nn
from torch import cuda

import hiervae
import opts
import hierdata
import Loss
import Trainer
import Optim
import pandas as pd
import sys

parser = argparse.ArgumentParser(description='train.py')

# opts.py
opts.add_md_help_argument(parser)
opts.model_opts(parser)
opts.train_opts(parser)

opt = parser.parse_args()
if opt.word_vec_size != -1:
    opt.src_word_vec_size = opt.word_vec_size
    opt.tgt_word_vec_size = opt.word_vec_size

if opt.layers != -1:
    opt.enc_layers = opt.layers
    opt.dec_layers = opt.layers

opt.brnn = (opt.encoder_type == "brnn")
if opt.seed > 0:
    torch.manual_seed(opt.seed)

if opt.rnn_type == "SRU" and not opt.gpuid:
    raise AssertionError("Using SRU requires -gpuid set.")

if torch.cuda.is_available() and not opt.gpuid:
    print("WARNING: You have a CUDA device, should run with -gpuid 0")

if opt.gpuid:
    cuda.set_device(opt.gpuid[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)



def report_func(epoch, batch, num_batches,
                start_time, lr, report_stats):
    """
    This is the user-defined batch-level traing progress
    report function.
    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): a Statistics instance.
    """
    if batch % opt.report_every == -1 % opt.report_every:
        report_stats.output(epoch, batch+1, num_batches, start_time)
        if opt.exp_host:
            report_stats.log("progress", experiment, lr)

def train_vae(model, train_iter, valid_iter, tgtvocab, optim):

    #train_iter = make_train_data_iter(train_data, opt)
    #valid_iter = make_valid_data_iter(valid_data, opt)


    train_loss = Loss.VaeLossCompute(model.generator, model.encoder, tgtvocab)
    valid_loss = Loss.VaeLossCompute(model.generator, model.encoder, tgtvocab)

    if use_gpu(opt):
        train_loss=train_loss.cuda()
        valid_loss=valid_loss.cuda()

    trunc_size = opt.truncated_decoder  # Badly named... default=0
    shard_size = opt.max_generator_batches #default=32

    trainer = Trainer.VaeTrainer(model, train_iter, valid_iter,
                           train_loss, valid_loss, optim,
                           trunc_size, shard_size)

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        # 1. Train for one epoch on the training set.
        train_stats = trainer.train(epoch, report_func)

        print('Train perplexity: %g' % train_stats.ppl())
        print('Train accuracy: %g' % train_stats.accuracy())

        # 2. Validate on the validation set.
        valid_stats = trainer.validate()
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation accuracy: %g' % valid_stats.accuracy())

        # 3. Log to remote server.
        if opt.exp_host:
            train_stats.log("train", experiment, optim.lr)
            valid_stats.log("valid", experiment, optim.lr)

        # 4. Update the learning rate
        trainer.epoch_step(valid_stats.ppl(), epoch)

        # 5. Drop a checkpoint if needed.
        if epoch >= opt.start_checkpoint_at:
            trainer.drop_checkpoint(opt, epoch, fields, valid_stats)

        train_loss.VAE_weightaneal(epoch)
        valid_loss.VAE_weightaneal(epoch)
        model.encoder.Varianceanneal()



def check_save_model_path():
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    print('encoder: ', enc)
    print('decoder: ', dec)




def build_optim(model, checkpoint):
    if opt.train_from:
        print('Loading optimizer from checkpoint.')
        optim = checkpoint['optim']
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())
    else:
        # what members of opt does Optim need?
        optim = Optim.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            opt=opt
        )
    optim.set_parameters(model.parameters())
    return optim


def main():
    train = pd.read_json('/D/home/lili/mnt/DATA/convaws/convdata/conv-train_v.json')
    val = pd.read_json('/D/home/lili/mnt/DATA/convaws/convdata/conv-val_v.json')
    train_srs = train.context.values.tolist()
    train_tgt = train.replies.values.tolist()
    val_srs = val.context.values.tolist()
    val_tgt = val.replies.values.tolist()

    '''
    test =  hierdata.buildvocab(train_tgt)
    print('vocab built')
    test2 =  hierdata.buildvocab(train_srs)
    sys.exit()
    '''


    src_vocab = hierdata.buildvocab(train_srs+val_srs)
    print('vocab built')
    tgt_vocab = hierdata.buildvocab(train_tgt+val_tgt)
    sys.exit()


    train_iter = hierdata.gen_minibatch(train_srs, train_tgt, src_vocab, tgt_vocab, mini_batch_size)
    valid_iter = hierdata.gen_minibatch(val_srs, val_tgt, src_vocab, tgt_vocab, test_batch_size)

    print('Building model...')
    model = hiervae.make_base_model(model_opt, src_vocab, tgt_vocab, use_gpu(opt), checkpoint) ### Done  #### How to integrate the two embedding layers...
    print(model)
    tally_parameters(model)### Done 
    check_save_model_path() ### Done

    # Build optimizer.
    optim = build_optim(model, checkpoint)

    # Do training.
    train_vae(model, train_iter, valid_iter, tgtvocab, optim)


if __name__ == "__main__":
    main()
