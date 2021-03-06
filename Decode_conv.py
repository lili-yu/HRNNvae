from __future__ import division

import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import cuda
import pandas as pd

import ModelHVAE
import opts
import data_util
import Loss
import Trainer
import Optim
import sys
import ujson
import json

print('starting')


parser = argparse.ArgumentParser(description='translate.py')
opts.add_md_help_argument(parser)

parser.add_argument('-model', 
                    help='Path to model .pt file')
parser.add_argument('-src',   required=False,
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

class GreedyDecoder(object):
    """Beam Search decoder."""

    def __init__(self, model,test_iter, src_vocab, tgt_vocab, tgtwords ):
        """Initialize model."""
        self.model = model
        self.iter = test_iter
        self.src_dict = src_vocab
        self.tgt_dict = tgt_vocab
        self.tgt_w = tgtwords 
        #self.config = config

    def decode_minibatch(
        self,
        input_lines_src,
        input_lines_trg
    ):
        """Decode a minibatch."""
        #self.config['data']['max_trg_length']
        for i in range(50):

            outputs,  dec_state, mu, logvar= \
                self.model(input_lines_src, input_lines_trg)
            word_probs = self.model.generator(outputs)
            #should have size of tgt_legth(=1), batchsize, vocabsize

            #decoder_logit = self.model(input_lines_src, input_lines_trg)
            #word_probs = self.model.decode(decoder_logit)
            decoder_argmax = word_probs.data.cpu().numpy().argmax(axis=-1)
            #size of tgt_legth(=1), batchsize
            #print(decoder_argmax.size())
            next_preds = Variable(
                torch.from_numpy(decoder_argmax[-1, :])
            ).cuda()

            input_lines_trg = torch.cat(
                (input_lines_trg, next_preds.unsqueeze(0)),
                0
            )

        return input_lines_trg

    def translate(self):
        """Evaluate model."""
        preds = []
        ground_truths = []
        for batch in self.iter:
            input_lines_src = batch[0]
            batchsize=input_lines_src.data.size()[1]
            input_lines_trg_gold = batch[1]

            input_lines_src = Variable(input_lines_src.data, volatile=True)
            #output_lines_src = Variable(output_lines_src.data, volatile=True)
            input_lines_trg_gold = Variable(input_lines_trg_gold.data, volatile=True)
            #sent_length, batchsize
            #output_lines_trg_gold = Variable(output_lines_trg_gold.data, volatile=True)


            input_lines_trg = Variable(torch.LongTensor([self.tgt_dict['<agent__ ']  for i in range(batchsize)]
            ), volatile=True).cuda()
            input_lines_trg = input_lines_trg.unsqueeze(0)
            print(input_lines_trg.size())

            '''
            for j in range(
            0, len(self.src),
             self.config['data']['batch_size']):

            print 'Decoding : %d out of %d ' % (j, len(self.src))
            # Get source minibatch
            input_lines_src, output_lines_src, lens_src, mask_src = (
                get_minibatch(
                    self.src['data'], self.src['word2id'], j,
                    self.config['data']['batch_size'],
                    self.config['data']['max_src_length'],
                    add_start=True, add_end=True
                )
            )

            input_lines_src = Variable(input_lines_src.data, volatile=True)
            output_lines_src = Variable(output_lines_src.data, volatile=True)

            # Get target minibatch
            input_lines_trg_gold, output_lines_trg_gold, lens_src, mask_src = (
                get_minibatch(
                    self.trg['data'], self.trg['word2id'], j,
                    self.config['data']['batch_size'],
                    self.config['data']['max_trg_length'],
                    add_start=True, add_end=True
                )
            )

            input_lines_trg_gold = Variable(input_lines_trg_gold.data, volatile=True)
            output_lines_trg_gold = Variable(output_lines_trg_gold.data, volatile=True)
            mask_src = Variable(mask_src.data, volatile=True)
            

            # Initialize target with <s> for every sentence
            input_lines_trg = Variable(torch.LongTensor(
                [
                    [trg['word2id']['<s>']]
                    for i in xrange(input_lines_src.size(0))
                ]
            ), volatile=True).cuda()
            '''

            # Decode a minibatch greedily __TODO__ add beam search decoding
            input_lines_trg = self.decode_minibatch(
                input_lines_src, input_lines_trg)

            # Copy minibatch outputs to cpu and convert ids to words
            input_lines_trg = input_lines_trg.data.cpu().numpy().transpose()
            input_lines_trg = [
                [self.tgt_w[x] for x in line]
                for line in input_lines_trg
            ]
            print('\ntranslated')
            print(' '.join(input_lines_trg[1]))
            #print(' '.join(input_lines_trg[2]))

            real = input_lines_trg_gold.data.cpu().numpy().transpose()
            real = [
                [self.tgt_w[x] for x in line]
                for line in real]
            print("real:")
            print(' '.join(real[1]))
            #print(' '.join(real[2]))
            '''

            # Do the same for gold sentences
            output_lines_trg_gold = output_lines_trg_gold.data.cpu().numpy().transpose()
            output_lines_trg_gold = [
                [self.tgt_w[x] for x in line]
                for line in output_lines_trg_gold
            ]

            # Process outputs
            for sentence_pred, sentence_real, sentence_real_src in zip(
                input_lines_trg,
                output_lines_trg_gold,
                output_lines_src
            ):
                if '</s>' in sentence_pred:
                    index = sentence_pred.index('</s>')
                else:
                    index = len(sentence_pred)
                preds.append(['<s>'] + sentence_pred[:index + 1])

                if '</s>' in sentence_real:
                    index = sentence_real.index('</s>')
                else:
                    index = len(sentence_real)

                ground_truths.append(['<s>'] + sentence_real[:index + 1])

        bleu_score = get_bleu(preds, ground_truths)
        print 'BLEU score : %.5f ' % (bleu_score)
        '''

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
    rebuild_vocab = False
    if rebuild_vocab:
        trainfile='/D/home/lili/mnt/DATA/convaws/convdata/conv-test_v.json'
        train = pd.read_json(trainfile)
        print('Read training data from: {}'.format(trainfile))

        valfile='/D/home/lili/mnt/DATA/convaws/convdata/conv-val_v.json'
        val = pd.read_json(valfile)
        print('Read validation data from: {}'.format(valfile))
        train_srs = train.context.values.tolist()
        train_tgt = train.replies.values.tolist()
        val_srs = val.context.values.tolist()
        val_tgt = val.replies.values.tolist()
        src_vocab, _ = hierdata.buildvocab(train_srs+val_srs)
        tgt_vocab, tgtwords = hierdata.buildvocab(train_tgt+val_tgt)

    else:
        print('load vocab from pt file')
        dicts = torch.load('test_vocabs.pt')
        #tgt = pd.read_json('./tgt.json')
        #src = pd.read_json('./src.json')
        src_vocab = dicts['src_word2id']
        tgt_vocab = dicts['tgt_word2id']
        tgtwords = dicts['tgt_id2word']
        print('source vocab size: {}'.format(len(src_vocab)))
        print('source vocab test, bill: {} , {}'.format(src_vocab['<pad>'],src_vocab['bill'] ))
        print('target vocab size: {}'.format(len(tgt_vocab)))
        print('target vocab test, bill: {}, {}'.format(tgt_vocab['<pad>'],tgt_vocab['bill'] ))
        print('target vocat testing:')
        print('word: <pad> get :{}'.format(tgtwords[tgt_vocab['<pad>']]))
        print('word: bill get :{}'.format(tgtwords[tgt_vocab['bill']]))
        print('word: service get :{}'.format(tgtwords[tgt_vocab['service']]))

    
    parser = argparse.ArgumentParser(description='train.py')

    # opts.py
    opts.add_md_help_argument(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)
    opt = parser.parse_args()


    dummy_opt = parser.parse_known_args([])[0]

    opt.cuda = opt.gpuid[0] > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpuid[0])

    checkpoint = opt.model
    print('Building model...')
    model = ModelHVAE.make_base_model(opt, src_vocab, tgt_vocab, opt.cuda, checkpoint) ### Done  #### How to integrate the two embedding layers...
    print(model)
    tally_parameters(model)### Done 


    testfile='/D/home/lili/mnt/DATA/convaws/convdata/conv-val_v.json'
    test = pd.read_json(testfile)
    print('Test training data from: {}'.format(testfile))

    test_srs = test.context.values.tolist()
    test_tgt = test.replies.values.tolist()

    test_batch_size = 16
    test_iter = data_util.gen_minibatch(test_srs, test_tgt,  test_batch_size, src_vocab, tgt_vocab)


    tgtvocab = tgt_vocab
    
    optim = Optim.Optim(
            'adam', 1e-3, 5
        )
    train_loss = Loss.VAELoss(model.generator,  tgtvocab)
    valid_loss = Loss.VAELoss(model.generator,  tgtvocab)
    trainer = Trainer.VaeTrainer(model, test_iter, test_iter,
                       train_loss, valid_loss, optim)
    valid_stats = trainer.validate()
    print('Validation perplexity: %g' % valid_stats.ppl())
    print('Validation accuracy: %g' % valid_stats.accuracy())


    # Do training.
    #decoder = GreedyDecoder(model,test_iter, src_vocab, tgt_vocab, tgtwords ) #model,test_iter,test_tgt, src_vocab, tgt_vocab
    #decoder.translate()


if __name__ == "__main__":
    main()
