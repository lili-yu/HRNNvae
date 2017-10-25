"""Decode Seq2Seq model with beam search."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model import Seq2Seq, Seq2SeqAttention
from data_utils import read_nmt_data, get_minibatch, read_config
from beam_search import Beam
from evaluate import get_bleu




class GreedyDecoder(object):
    """Beam Search decoder."""

    def __init__(self,model,test_srs, src_vocab, tgt_vocab):
        """Initialize model."""
        self.src = test_srs
        self.trg = test_tgt
        self.src_dict = src_vocab
        self.tgt_dict = tgt_vocab
        self.model = model

    def decode_minibatch(
        self,
        input_lines_src,
        input_lines_trg,
        output_lines_trg_gold
    ):
        """Decode a minibatch."""
        for i in xrange(self.config['data']['max_trg_length']):

            decoder_logit = self.model(input_lines_src, input_lines_trg)
            word_probs = self.model.decode(decoder_logit)
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
        for j in xrange(
            0, len(self.src['data']),
            self.config['data']['batch_size']
        ):

            print 'Decoding : %d out of %d ' % (j, len(self.src['data']))
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
            mask_src = Variable(mask_src.data, volatile=True)

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

            # Decode a minibatch greedily __TODO__ add beam search decoding
            input_lines_trg = self.decode_minibatch(
                input_lines_src, input_lines_trg,
                output_lines_trg_gold
            )

            # Copy minibatch outputs to cpu and convert ids to words
            input_lines_trg = input_lines_trg.data.cpu().numpy()
            input_lines_trg = [
                [self.trg['id2word'][x] for x in line]
                for line in input_lines_trg
            ]

            # Do the same for gold sentences
            output_lines_trg_gold = output_lines_trg_gold.data.cpu().numpy()
            output_lines_trg_gold = [
                [self.trg['id2word'][x] for x in line]
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

if __name__ == '__main__':

    model_config = '/home/sandeep/Research/nmt-pytorch/config_local_en_de_attention_wmt15.json'
    model_weights = '/home/sandeep/Models/torch_seq2seq/model_translation__src_en__trg_de__attention_attention__dim_1024__emb_dim_500__optimizer_adam__n_layers_src_2__n_layers_trg_1__bidir_True__epoch_6.model'

    config = read_config(model_config)

    src, trg = read_nmt_data(
        src=config['data']['src'],
        config=config,
        trg=config['data']['trg']
    )

    src_test, trg_test = read_nmt_data(
        src=config['data']['test_src'],
        config=config,
        trg=config['data']['test_trg']
    )

    src_test['word2id'] = src['word2id']
    src_test['id2word'] = src['id2word']

    trg_test['word2id'] = trg['word2id']
    trg_test['id2word'] = trg['id2word']

    # decoder = BeamSearchDecoder(config, model_weights, src_test, trg_test)
    # decoder.translate()

    decoder = GreedyDecoder(config, model_weights, src_test, trg_test)
    decoder.translate()
    '''
    allHyp, allScores = decoder.decode_batch(0)
    all_hyp_inds = [[x[0] for x in hyp] for hyp in allHyp]
    all_preds = [' '.join([trg['id2word'][x] for x in hyp]) for hyp in all_hyp_inds]

    input_lines_trg_gold, output_lines_trg_gold, lens_src, mask_src = (
        get_minibatch(
            trg['data'], trg['word2id'], 0,
            80,
            50,
            add_start=True, add_end=True
        )
    )

    output_lines_trg_gold = output_lines_trg_gold.data.cpu().numpy()
    all_gold_inds = [[x for x in hyp] for hyp in output_lines_trg_gold]
    all_gold = [' '.join([trg['id2word'][x] for x in hyp]) for hyp in all_gold_inds]

    for hyp, gt in zip(all_preds, all_gold):
        print hyp, len(hyp.split())
        print '-------------------------------------------------'
        print gt
        print '================================================='
    '''
