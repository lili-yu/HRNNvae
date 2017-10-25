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
                self.model(input_lines_src, input_lines_trg, dec_state)
            word_probs = self.model.generator(outputs.view(-1, outputs.size(2)))

            #decoder_logit = self.model(input_lines_src, input_lines_trg)
            #word_probs = self.model.decode(decoder_logit)
            decoder_argmax = word_probs.data.cpu().numpy().argmax(axis=-1)
            next_preds = Variable(
                torch.from_numpy(decoder_argmax[:, -1])
            ).cuda()

            input_lines_trg = torch.cat(
                (input_lines_trg, next_preds.unsqueeze(1)),
                1
            )

        return input_lines_trg

    def translate(self):
        """Evaluate model."""
        preds = []
        ground_truths = []
        for batch in self.iter:
            input_lines_src = batch[0]
            input_lines_trg_gold = batch[1]

            input_lines_src = Variable(input_lines_src.data, volatile=True)
            #output_lines_src = Variable(output_lines_src.data, volatile=True)
            input_lines_trg_gold = Variable(input_lines_trg_gold.data, volatile=True)
            #output_lines_trg_gold = Variable(output_lines_trg_gold.data, volatile=True)

            input_lines_trg = Variable(torch.LongTensor(
                [
                    [trg['word2id']['<agent__']]
                    for i in xrange(input_lines_src.size(0))
                ]
            ), volatile=True).cuda()

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
            input_lines_trg = input_lines_trg.data.cpu().numpy()
            input_lines_trg = [
                [self.tgt_w[x] for x in line]
                for line in input_lines_trg
            ]

            print(input_lines_trg)

            '''

            # Do the same for gold sentences
            output_lines_trg_gold = output_lines_trg_gold.data.cpu().numpy()
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

    testfile='/D/home/lili/mnt/DATA/convaws/convdata/conv-test_v.json'
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

    #config = read_config(model_config)

    test_batch_size = 16
    test_iter = hierdata.gen_minibatch(test_srs, test_tgt,  test_batch_size, src_vocab, tgt_vocab)

    checkpoint = opt.model
    print('Building model...')
    model = hiervae.make_base_model(opt, src_vocab, tgt_vocab, use_gpu(opt), checkpoint) ### Done  #### How to integrate the two embedding layers...
    print(model)
    tally_parameters(model)### Done 

    # Do training.
    decoder = GreedyDecoder(model,test_iter, src_vocab, tgt_vocab, tgtwords ) #model,test_iter,test_tgt, src_vocab, tgt_vocab
    decoder.translate()


if __name__ == "__main__":
    main()
