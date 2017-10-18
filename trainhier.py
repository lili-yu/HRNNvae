from __future__ import division

import os
import argparse
import torch
import torch.nn as nn
from torch import cuda

import hiervae
import opts
from hierdata import *
from embeddings


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



def train_vae(model, train_iter, valid_iter, tgtvocab, optim):

    #train_iter = make_train_data_iter(train_data, opt)
    #valid_iter = make_valid_data_iter(valid_data, opt)


    train_loss = onmt.Loss.VaeLossCompute(model.generator, model.encoder, tgtvocab)
    valid_loss = onmt.Loss.VaeLossCompute(model.generator, model.encoder, tgtvocab)

    if use_gpu(opt):
        train_loss=train_loss.cuda()
        valid_loss=valid_loss.cuda()

    trunc_size = opt.truncated_decoder  # Badly named... default=0
    shard_size = opt.max_generator_batches #default=32

    trainer = onmt.VaeTrainer(model, train_iter, valid_iter,
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






def build_model(model_opt, opt, fields, checkpoint):
    print('Building model...')
    model = hiervae.make_base_model(model_opt, fields,
                                                  use_gpu(opt), checkpoint)
    if len(opt.gpuid) > 1:
        print('Multi gpu training ', opt.gpuid)
        model = nn.DataParallel(model, device_ids=opt.gpuid, dim=1)
    print(model)
    return model


def build_optim(model, checkpoint):
    if opt.train_from:
        print('Loading optimizer from checkpoint.')
        optim = checkpoint['optim']
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())
    else:
        # what members of opt does Optim need?
        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            opt=opt
        )
    optim.set_parameters(model.parameters())
    return optim


def main():
    train = pd.read_json('imdb_final.json')
    test = pd.read_json('imdb_final.json')
    train_srs = train.context
    train_tgt = train.replies
    val_srs = train.context
    val_tgt = train.replies

    emb_layer_srs = embeddings.EmbeddingLayer(
        args.d, train_srs+val_srs,
        embs = dataloader.load_embedding(args.embedding)
    )


    train_iter = gen_minibatch(train_srs, train_tgt, emb_layer_srs.word2id, mini_batch_size)
    valid_iter = gen_minibatch(val_srs, val_tgt, emb_layer_srs.word2id, test_batch_size)

    model = build_model(model_opt, opt, fields, checkpoint) ### Done
    tally_parameters(model)### Done
    check_save_model_path() ### Done

    # Build optimizer.
    optim = build_optim(model, checkpoint)

    # Do training.
    train_vae(model, train_iter, valid_iter, tgtvocab, optim)


if __name__ == "__main__":
    main()
