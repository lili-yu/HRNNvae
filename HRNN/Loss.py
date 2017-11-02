"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
from Trainer import Statistics




class VAELoss(nn.Module):
    def __init__(self, generator, tgt_vocab):
        super(VAELoss, self).__init__()

        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_vocab['<pad>'] #tgt_vocab.stoi[onmt.IO.PAD_WORD]   ########To do 

        weight = torch.ones(len(tgt_vocab)).cuda()
        weight[self.padding_idx] = 0
        self.criterion = nn.NLLLoss(weight, size_average=False)

        self.kld_weight =  0.05 #0.05
        self.kld_start_inc = 15 #10000
        self.kld_max = 0.5
        self.kld_inc = 0.05
        #temperature = 0.9
        #temperature_min = 0.5
        #temperature_dec = 0.000002
        #self.kld_weight = 0.05

    
    def compute_loss(self, output, target, **kwargs):
        """ See base class for args description. """
        scores = self.generator(self.bottle(output))
        scores_data = scores.data.clone()

        #print(target[:,0] )
        #print(target[:,1] )
        target = target[1:]
        target = target.view(-1)
        target_data = target.data.clone()


        loss = self.criterion(scores, target)
        loss_data = loss.data.clone()

        '''print("scores: {}".format(scores[:10]))
        print("target: {}".format(target[:10]))
        print("loss: {}".format(loss))
        '''

        KLD_data = 0
        #print(type(KLD_data))

        stats = self.stats(loss_data, KLD_data, scores_data, target_data)

        return loss, stats

    def forward(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define the compute_loss().
        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs: additional info for computing loss.
        """
        # Need to simplify this interface.
        return self.compute_loss(batch, output, target, **kwargs)

    def stats(self, loss, KLD, scores, target):
        """
        Compute and return a Statistics object.

        Args:
            loss(Tensor): the loss computed by the loss criterion.
            scores(Tensor): a sequence of predict output with scores.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)

        '''
        print('soresize: {}'.format(scores.size()))
        print('pred: {}'.format(pred.size()))
        print('target: {}'.format(target.size()))
        print('target: {}'.format(target[-100:-80]))
        print(non_padding[-100:-80])
        
        print('\n sampling the result')
        print('pred: {}'.format(pred[-100:-80]))
        print('target: {}'.format(target[-100:-80]))
        '''

        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum()
        #print('words: {}'.format(non_padding.sum()))
        #print('correct: {}'.format(num_correct))
        return Statistics(loss[0], KLD, non_padding.sum(), num_correct)

    def bottle(self, v):
        return v.view(-1, v.size(2))

    def unbottle(self, v, batch_size):
        return v.view(-1, batch_size, v.size(1))



    def VAE_weightaneal(self, epoch):
        """ See base class for args description. """
        if epoch*5 >= self.kld_start_inc and self.kld_weight < self.kld_max:
            self.kld_weight += self.kld_inc
            print(("kld_weight updated to: %6.3f") %(self.kld_weight)) 

