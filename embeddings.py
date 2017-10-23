import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

def deep_iter(x):
    if isinstance(x, list) or isinstance(x, tuple):
        for u in x:
            for v in deep_iter(u):
                yield v
    else:
        yield x


class EmbeddingLayer(nn.Module):
    def __init__(self, n_d,  vocab=None, words=None, embs=None, fix_emb=True, oov='<oov>', pad='<pad>', normalize=False):
        super(EmbeddingLayer, self).__init__()
        
        word2id = {}
        if embs is not None:
            embwords, embvecs = embs
            for word in embwords:
                assert word not in word2id, "Duplicate words in pre-trained embeddings"
                word2id[word] = len(word2id)

            sys.stdout.write("{} pre-trained word embeddings loaded.\n".format(len(word2id)))
            if n_d != len(embvecs[0]):
                sys.stdout.write("[WARNING] n_d ({}) != word vector size ({}). Use {} for embeddings.\n".format(
                    n_d, len(embvecs[0]), len(embvecs[0])
                ))
                n_d = len(embvecs[0])

        elif vocab is not None:
            word2id = vocab

        else:
            for w in deep_iter(words):
                if w not in word2id:
                    word2id[w] = len(word2id)

        if oov not in word2id:
            word2id[oov] = len(word2id)

        if pad not in word2id:
            word2id[pad] = len(word2id)

        self.word2id = word2id
        self.n_V, self.n_d = len(word2id), n_d
        self.oovid = word2id[oov]
        self.padid = word2id[pad]
        self.embedding = nn.Embedding(self.n_V, n_d)
        self.embedding.weight.data.uniform_(-0.25, 0.25)

        if embs is not None:
            weight  = self.embedding.weight
            weight.data[:len(embwords)].copy_(torch.from_numpy(embvecs))
            sys.stdout.write("embedding shape: {}\n".format(weight.size()))

        if normalize:
            weight = self.embedding.weight
            norms = weight.data.norm(2,1)
            if norms.dim() == 1:
                norms = norms.unsqueeze(1)
            weight.data.div_(norms.expand_as(weight.data))

        if fix_emb:
            self.embedding.weight.requires_grad = False

    def forward(self, input):
        return self.embedding(input)
