import torch
import torch.nn as nn
from torch.autograd import Variable



class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.arange(0, max_len).unsqueeze(1).expand(max_len, dim)
        div_term = 1 / torch.pow(10000, torch.arange(0, dim * 2, 2) / dim)
        pe = pe * div_term.expand_as(pe)
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, emb):
        # We must wrap the self.pe in Variable to compute, not the other
        # way - unwrap emb(i.e. emb.data). Otherwise the computation
        # wouldn't be watched to build the compute graph.
        emb = emb + Variable(self.pe[:emb.size(0), :1, :emb.size(2)]
                             .expand_as(emb), requires_grad=False)
        emb = self.dropout(emb)
        return emb


class EmbeddingLayer(nn.Module):
    """
    Words embeddings dictionary for encoder/decoder.

    Args:
        word_vec_size (int): size of the dictionary of embeddings.
        position_encoding (bool): use a sin to mark relative words positions.
        feat_merge (string): merge action for the features embeddings:
                    concat, sum or mlp.
        feat_vec_exponent (float): when using '-feat_merge concat', feature
                    embedding size is N^feat_dim_exponent, where N is the
                    number of values of feature takes.
        feat_vec_size (int): embedding dimension for features when using
                    '-feat_merge mlp'
        dropout (float): dropout probability.
        word_padding_idx (int): padding index for words in the embeddings.
        feats_padding_idx ([int]): padding index for a list of features
                                   in the embeddings.
        word_vocab_size (int): size of dictionary of embeddings for words.
        feat_vocab_sizes ([int], optional): list of size of dictionary
                                    of embeddings for each feature.
    """
    def __init__(self, word_vec_size,  vocab):

        word_padding_idx = 0
        self.word_padding_idx = vocab['<pad>']
        word_vocab_size = len(vocab)

        # Dimensions and padding for constructing the word embedding matrix
        vocab_sizes = [word_vocab_size]
        emb_dims = [word_vec_size]
        pad_indices = [word_padding_idx]

        # Dimensions and padding for feature embedding matrices
        # (these have no effect if feat_vocab_sizes is empty)
        '''
        if feat_merge == 'sum':
            feat_dims = [word_vec_size] * len(feat_vocab_sizes)
        elif feat_vec_size > 0:
            feat_dims = [feat_vec_size] * len(feat_vocab_sizes)
        else:
            feat_dims = [int(vocab ** feat_vec_exponent)
                         for vocab in feat_vocab_sizes]

        vocab_sizes.extend(feat_vocab_sizes)
        emb_dims.extend(feat_dims)
        pad_indices.extend(feat_padding_idx)
        '''

        # The embedding matrix look-up tables. The first look-up table
        # is for words. Subsequent ones are for features, if any exist.
        emb_params = zip(vocab_sizes, emb_dims, pad_indices)
        self.vocab_sizes = vocab_sizes 
        embeddings = [nn.Embedding(vocab, dim, padding_idx=pad)
                      for vocab, dim, pad in emb_params]
        emb_luts = embeddings #Elementwise(feat_merge, embeddings)

        # The final output size of word + feature vectors. This can vary
        # from the word vector size if and only if features are defined.
        # This is the attribute you should access if you need to know
        # how big your embeddings are going to be.
        self.embedding_size = word_vec_size


        # The sequence of operations that converts the input sequence
        # into a sequence of embeddings. At minimum this consists of
        # looking up the embeddings for each word and feature in the
        # input. Model parameters may require the sequence to contain
        # additional operations as well.
        super(EmbeddingLayer, self).__init__()
        self.make_embedding = nn.Sequential()
        self.make_embedding.add_module('emb_luts', emb_luts)

    @property
    def word_lut(self):
        return self.make_embedding[0][0]

    @property
    def emb_luts(self):
        return self.make_embedding[0]

    def load_pretrained_vectors(self, emb_file, fixed):
        if emb_file:
            pretrained = torch.load(emb_file)
            self.word_lut.weight.data.copy_(pretrained)
            if fixed:
                self.word_lut.weight.requires_grad = False

    def forward(self, input):
        """
        Return the embeddings for words, and features if there are any.
        Args:
            input (LongTensor): len x batch x nfeat
        Return:
            emb (FloatTensor): len x batch x self.embedding_size
        """
        in_length, in_batch, nfeat = input.size()
        aeq(nfeat, len(self.emb_luts))

        emb = self.make_embedding(input)

        out_length, out_batch, emb_size = emb.size()
        aeq(in_length, out_length)
        aeq(in_batch, out_batch)
        aeq(emb_size, self.embedding_size)

        return emb



#########====================================================================##############
def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)



class Bottle(nn.Module):
        def forward(self, input):
            if len(input.size()) <= 2:
                return super(Bottle, self).forward(input)
            size = input.size()[:2]
            out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
            return out.contiguous().view(size[0], size[1], -1)


class Bottle2(nn.Module):
        def forward(self, input):
            if len(input.size()) <= 3:
                return super(Bottle2, self).forward(input)
            size = input.size()
            out = super(Bottle2, self).forward(input.view(size[0]*size[1],
                                                          size[2], size[3]))
            return out.contiguous().view(size[0], size[1], size[2], size[3])


class LayerNorm(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNorm, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z
        mu = torch.mean(z, dim=1)
        sigma = torch.std(z, dim=1)
        # HACK. PyTorch is changing behavior
        if mu.dim() == 1:
            mu = mu.unsqueeze(1)
            sigma = sigma.unsqueeze(1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out.mul(self.a_2.expand_as(ln_out)) \
            + self.b_2.expand_as(ln_out)
        return ln_out


class BottleLinear(Bottle, nn.Linear):
    pass


class BottleLayerNorm(Bottle, LayerNorm):
    pass


class BottleSoftmax(Bottle, nn.Softmax):
    pass


class Elementwise(nn.ModuleList):
    """
    A simple network container.
    Parameters are a list of modules.
    Inputs are a 3d Variable whose last dimension is the same length
    as the list.
    Outputs are the result of applying modules to inputs elementwise.
    An optional merge parameter allows the outputs to be reduced to a
    single Variable.
    """

    def __init__(self, merge=None, *args):
        assert merge in [None, 'first', 'concat', 'sum', 'mlp']
        self.merge = merge
        super(Elementwise, self).__init__(*args)

    def forward(self, input):
        inputs = [feat.squeeze(2) for feat in input.split(1, dim=2)]
        assert len(self) == len(inputs)
        outputs = [f(x) for f, x in zip(self, inputs)]
        if self.merge == 'first':
            return outputs[0]
        elif self.merge == 'concat' or self.merge == 'mlp':
            return torch.cat(outputs, 2)
        elif self.merge == 'sum':
            return sum(outputs)
        else:
            return outputs

