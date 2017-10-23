from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
#from torch.nn.utils.rnn import pack_padded_sequence as pack
#from torch.nn.utils.rnn import pad_packed_sequence as unpack


import sys
import numpy as np
import SRU
import embeddings

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def use_gpu(opt):
    return (hasattr(opt, 'gpuid') and len(opt.gpuid) > 0) or \
        (hasattr(opt, 'gpu') and opt.gpu > -1)

################################################# Encoder  #################################################
class wordEncoder(nn.Module):

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout, embeddings, z_size):
        super(wordEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings

        if rnn_type == "SRU":
            # SRU doesn't support PackedSequence.
            self.wordrnn = SRU.SRU(
                    input_size=embeddings.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)
        else:
            self.wordrnn = getattr(nn, rnn_type)(
                    input_size=embeddings.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)


    def forward(self, input, hidden=None):
        """
        Args:
            input (LongTensor): len x batch x nfeat.
            lengths (LongTensor): batch
            hidden: Initial hidden state.
        Returns:
            hidden_t (Variable): Pair of layers x batch x rnn_size - final
                                    encoder state
            outputs (FloatTensor):  len x batch x rnn_size -  Memory bank
        """
        """ See EncoderBase.forward() for description of args and returns."""

        emb = self.embeddings(input)
        s_len, batch, emb_dim = emb.size()

        outputs, hidden_t = self.wordrnn(emb, hidden)
        return hidden_t, outputs



class ConvEncoder(nn.Module):

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout, embeddings, z_size):
        super(ConvEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions

        self.wordrnn = wordEncoder(rnn_type, bidirectional, num_layers, hidden_size, dropout, embeddings)
        if rnn_type == "SRU":
            # SRU doesn't support PackedSequence.
            self.convrnn = SRU.SRU(
                    input_size=embeddings.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)
        else:
            self.convrnn = getattr(nn, rnn_type)(
                    input_size=embeddings.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)
        self.h2z = nn.Linear(hidden_size, z_size * 2)

        self.varcoeff = 0.0
        self.varstep = 0.1

    def forward(self, input, state_word=None, hidden_sent=None):
        """
        Args:
            input (LongTensor): len x batch x nfeat.
            lengths (LongTensor): batch
            hidden: Initial hidden state.
        Returns:
            hidden_t (Variable): Pair of layers x batch x rnn_size - final
                                    encoder state
            outputs (FloatTensor):  len x batch x rnn_size -  Memory bank
        """
        """ See EncoderBase.forward() for description of args and returns."""
        max_sents, batch_size, max_tokens = input.size()
        s = None
        for i in xrange(max_sents):
            _s, state_word = self.wordrnn(input[i,:,:].transpose(0,1), state_word)
            if(s is None):
                s = _s
            else:
                s = torch.cat((s,_s),0)      

        outputs_sent, hidden_sent  = self.convrnn(s, hidden_sent )

        #new_h = [torch.cat([hidden_t[i][0:h.size(0):2], hidden_t[i][1:h.size(0):2]], 2) for i in range(len(hidden_t))]
        hh = hidden_sent [0] #in LSTM, hidden_t[0] is hidden states, while hidden_t[1] is cell states
        #hh.size(), num_layers * num_directions, batch, hidden_size
        hhh = torch.mean(hh,0)

        ps = self.h2z(hhh)
        mu, logvar = torch.chunk(ps, 2, dim=1)
        z = self.sample(mu, logvar)

        #return hidden_t, outputs
        #return mu, logvar, z, hidden_t, outputs
        return outputs, z,  mu, logvar

    def sample(self, mu, logvar):
        #print(mu.size())
        #print(logvar.size())
        eps = Variable(torch.randn(logvar.size())).cuda() #.cuda()
        #eps = eps.cuda()
        std = torch.exp(logvar / 2.0)
        z = mu + eps*std*self.varcoeff
        #print(z.size())
        return z #torch.mm(eps ,std)

    def Varianceanneal(self):
        if self.varcoeff <=1:
            self.varcoeff += self.varstep
            print(("Encoder variation scaling change to: %6.3f") %(self.varcoeff))



################################################# Decoder  #################################################

class RNNDecoder(nn.Module):
    """
    RNN decoder base class.
    """
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type, coverage_attn, context_gate,
                 copy_attn, dropout, embeddings):
        super(RNNDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)
        self.worddropout = nn.Dropout(dropout)

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type, self._input_size, hidden_size,
                                   num_layers, dropout)


    def forward(self, input, context, state):
        """
        Forward through the decoder.
        Args:
            input (LongTensor): a sequence of input tokens tensors
                                of size (len x batch x nfeats).
            context (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
        Returns:
            outputs (FloatTensor): a Tensor sequence of output from the decoder
                                   of shape (len x batch x hidden_size).
            state (FloatTensor): final hidden state from the decoder.
            attns (dict of (str, FloatTensor)): a dictionary of different
                                type of attention Tensor from the decoder
                                of shape (src_len x batch).
        """
        # Args Check
        assert isinstance(state, RNNDecoderState)
        input_len, input_batch, _ = input.size()
        contxt_len, contxt_batch, _ = context.size()
        aeq(input_batch, contxt_batch)
        # END Args Check

        # Run the forward pass of the RNN.
        outputs = []
        '''
        input = input.type(torch.cuda.FloatTensor)
        mask = input.ge(0.3).type(torch.cuda.FloatTensor)
        input = input *mask
        input = input.type(torch.cuda.LongTensor)
        '''

        emb = self.embeddings(input)
        emb = self.worddropout(emb)
        #print(emb.size())

        #h0 = self._fix_enc_hidden(state.hidden[0])
        h0 =  state.hidden[0][0]
        #print(h0.size())
        h0_all = h0.repeat(emb.size()[0], 1,1)
        emb = torch.cat([emb, h0_all],2)
        #print(state.hidden[0].size())

        # Run the forward pass of the RNN.
        rnn_output, hidden = self.rnn(emb, state)
        outputs = self.dropout(rnn_output)

        # Result 
        input_len, input_batch, _ = input.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(input_len, output_len)
        aeq(input_batch, output_batch)

        # Update the state with the result.
        final_output = outputs[-1]
        #state.update_state(hidden, final_output.unsqueeze(0))

        # Concatenates sequence of tensors along a new dimension.
        outputs = torch.stack(outputs)
       
        return outputs, hidden  #, attns

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim.
        We need to convert it to layers x batch x (directions*dim).
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h


    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        """
        Private helper for building standard decoder RNN.
        """
        # Use pytorch version when available.
        if rnn_type == "SRU":
            return SRU.SRU(
                    input_size, hidden_size,
                    num_layers=num_layers,
                    dropout=dropout)

        return getattr(nn, rnn_type)(
            input_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout)

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embeddings.embedding_size + self.hidden_size




################################################# Generator  #################################################
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size ):
        super(Generator, self).__init__()
        self.temperature = 0.5
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size, hidden_size)
        self.out = nn.LogSoftmax()

    def forward(self, input):
        x = self.linear(input)
        x = x.div(self.temperature)
        x = self.out(x)
        return x

################################################# VAE  #################################################
class VaeModel(nn.Module):
    """
    The encoder + decoder Neural Machine Translation Model.
    """
    def __init__(self, encoder, decoder, model_opt, smultigpu=False):
        """
        Args:
            encoder(*Encoder): the various encoder.
            decoder(*Decoder): the various decoder.
            multigpu(bool): run parellel on multi-GPU?
        """
        super(VaeModel, self).__init__()
        z_size = model_opt.z_size
        hidden_size = model_opt.rnn_size
        self.encoder = encoder
        self.z2h = nn.Linear(z_size, hidden_size)
        self.z2c = nn.Linear(z_size, hidden_size)
        self.decoder = decoder
        self.n_layers = decoder.num_layers

    def forward(self, src, tgt, dec_state=None):
        """
        Args:
            src(FloatTensor): a sequence of source tensors with
                    optional feature tensors of size (len x batch).
            tgt(FloatTensor): a sequence of target tensors with
                    optional feature tensors of size (len x batch).
            lengths([int]): an array of the src length.
            dec_state: A decoder state object
        Returns:
            outputs (FloatTensor): (len x batch x hidden_size): decoder outputs
            attns (FloatTensor): Dictionary of (src_len x batch)
            dec_hid den (FloatTensor): tuple (1 x batch x hidden_size)
                                      Init hidden state
        """
        src = src
        tgt = tgt[:-1]  # exclude last target from inputs
        context, z,  mu, logvar = self.encoder(src)
        #enc_z = self.z2h(z)
        enc_z = self.z2h(z).unsqueeze(0).repeat(self.n_layers, 1, 1)
        cell_z = self.z2c(z).unsqueeze(0).repeat(self.n_layers, 1, 1)
        enc_state = self.decoder.init_decoder_state(context, (enc_z,cell_z))
        out, dec_state = self.decoder(tgt, context, enc_state)
        
        return out, dec_state, mu, logvar




################################################# make_function  #################################################


def make_base_model(model_opt, src_dict, tgt_dict, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """
    # Make encoder.
    src_embeddings = embeddings.EmbeddingLayer(model_opt.word_vec_size, vocab=src_dict) #,embs = dataloader.load_embedding(args.embedding))
    encoder = make_encoder(model_opt, src_embeddings)


    # Make decoder.
    src_embeddings =  embeddings.EmbeddingLayer(model_opt.word_vec_size, vocab=tgt_dict)
    decoder = make_decoder(model_opt, tgt_embeddings)

    # Make NMTModel(= encoder + decoder).
    #model = NMTModel(encoder, decoder)
    model = VaeModel(encoder, decoder, model_opt)
    
    generator = Generator(model_opt.rnn_size, len(tgt_dict)) 

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
    else:
        if model_opt.param_init != 0.0:
            print('Intializing parameters.')
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        model.encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
        model.decoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

    # add the generator to the module (does this register the parameter?)
    model.generator = generator

    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model
