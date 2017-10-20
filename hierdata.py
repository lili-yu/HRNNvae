
# coding: utf-8

# ## Boilerplate

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def deep_iter(x):
    if isinstance(x, list) or isinstance(x, tuple):
        for u in x:
            for v in deep_iter(u):
                yield v
    else:
        yield x

def buildvocab(words, oov='<oov>', pad='<pad>'):
    word2id = {}
    word2id[pad] = len(word2id)
    word2id[oov] = len(word2id)

    for w in deep_iter(words):
        if w not in word2id:
            word2id[w] = len(word2id)

    print("The vocab size is: {}".format(len(word2id)))
    return word2id




def pad_batch(context, src_vocab, reverse_pad = False):

    mini_batch_size = len(mini_batch)
    max_sent_len = int(np.mean([len(x) for x in mini_batch]))
    max_token_len = int(np.mean([len(val) for sublist in mini_batch for val in sublist]))
    main_matrix = np.zeros((mini_batch_size, max_sent_len, max_token_len), dtype= np.int)

    if not reverse_pad:
        for i in range(main_matrix.shape[0]):
            for j in range(main_matrix.shape[1]):
                for k in range(main_matrix.shape[2]):
                    try:
                        main_matrix[i,j,k] = src_vocab[mini_batch[i][j][k]]
                    except IndexError:
                        pass
    if reverse_pad:
        for i in range(main_matrix.shape[0]):
            for j in range(main_matrix.shape[1]):
                for k in range(main_matrix.shape[2]):
                    try:
                        main_matrix[-i-1,-j-1,-k-1] =src_vocab[mini_batch[-i-1][-j-1][-k-1]] 
                    except IndexError:
                        pass
    return Variable(torch.from_numpy(main_matrix).transpose(0,1))



def pad_batch_reply(reply_batch, tgt_vocab):
    
    mini_batch_size = len(mini_batch)
    max_sent_len = int(np.mean([len(x) for x in mini_batch]))
    main_matrix = np.zeros((mini_batch_size, max_sent_len, max_token_len), dtype= np.int)

    for i in range(main_matrix.shape[0]):
        for j in range(main_matrix.shape[1]):
                try:
                    main_matrix[i,j] = tgt_vocab[mini_batch[i][j]]
                except IndexError:
                    pass
    return Variable(torch.from_numpy(main_matrix).transpose(0,1))


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]



def gen_minibatch(srs, tgt, mini_batch_size, src_vocab, tgt_vocab, shuffle= True):
    for context, reply in iterate_minibatches(srs, tgt, mini_batch_size, shuffle= shuffle):
        contexts = pad_batch(context, src_vocab, reverse_pad = True)
        reply  = pad_batch_reply(reply , tgt_vocab)
        yield contexts.cuda(), reply.cuda()


'''
# In[9]:

def train_data(mini_batch, targets, word_attn_model, sent_attn_model, word_optimizer, sent_optimizer, criterion):
    state_word = word_attn_model.init_hidden().cuda()
    state_sent = sent_attn_model.init_hidden().cuda()
    max_sents, batch_size, max_tokens = mini_batch.size()
    word_optimizer.zero_grad()
    sent_optimizer.zero_grad()
    s = None
    for i in xrange(max_sents):
        _s, state_word, _ = word_attn_model(mini_batch[i,:,:].transpose(0,1), state_word)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)            
    y_pred, state_sent, _ = sent_attn_model(s, state_sent)
    loss = criterion(y_pred.cuda(), targets) 
    loss.backward()
    
    word_optimizer.step()
    sent_optimizer.step()
    
    return loss.data[0]


# In[10]:

def get_predictions(val_tokens, word_attn_model, sent_attn_model):
    max_sents, batch_size, max_tokens = val_tokens.size()
    state_word = word_attn_model.init_hidden().cuda()
    state_sent = sent_attn_model.init_hidden().cuda()
    s = None
    for i in xrange(max_sents):
        _s, state_word, _ = word_attn_model(val_tokens[i,:,:].transpose(0,1), state_word)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)            
    y_pred, state_sent, _ = sent_attn_model(s, state_sent)    
    return y_pred


# In[11]:




# ## Loading the data

# In[14]:

d = pd.read_json('/data1/sandeep/datasets/imdb_final.json')

d['rating'] = d['rating'] - 1

from sklearn.cross_validation import train_test_split

d = d[['tokens','rating']]

X = d.tokens
y = d.rating

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size = 0.3, random_state= 42)

y_train.shape





# In[22]:

def test_accuracy_mini_batch(tokens, labels, word_attn, sent_attn):
    y_pred = get_predictions(tokens, word_attn, sent_attn)
    _, y_pred = torch.max(y_pred, 1)
    correct = np.ndarray.flatten(y_pred.data.cpu().numpy())
    labels = np.ndarray.flatten(labels.data.cpu().numpy())
    num_correct = sum(correct == labels)
    return float(num_correct) / len(correct)


# In[23]:

def test_accuracy_full_batch(tokens, labels, mini_batch_size, word_attn, sent_attn):
    p = []
    l = []
    g = gen_minibatch(tokens, labels, mini_batch_size)
    for token, label in g:
        y_pred = get_predictions(token.cuda(), word_attn, sent_attn)
        _, y_pred = torch.max(y_pred, 1)
        p.append(np.ndarray.flatten(y_pred.data.cpu().numpy()))
        l.append(np.ndarray.flatten(label.data.cpu().numpy()))
    p = [item for sublist in p for item in sublist]
    l = [item for sublist in l for item in sublist]
    p = np.array(p)
    l = np.array(l)
    num_correct = sum(p == l)
    return float(num_correct)/ len(p)


# In[24]:

def test_data(mini_batch, targets, word_attn_model, sent_attn_model):    
    state_word = word_attn_model.init_hidden().cuda()
    state_sent = sent_attn_model.init_hidden().cuda()
    max_sents, batch_size, max_tokens = mini_batch.size()
    s = None
    for i in xrange(max_sents):
        _s, state_word, _ = word_attn_model(mini_batch[i,:,:].transpose(0,1), state_word)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)            
    y_pred, state_sent,_ = sent_attn_model(s, state_sent)
    loss = criterion(y_pred.cuda(), targets)     
    return loss.data[0]


# In[25]:


# In[27]:

def check_val_loss(val_tokens, val_labels, mini_batch_size, word_attn_model, sent_attn_model):
    val_loss = []
    for token, label in iterate_minibatches(val_tokens, val_labels, mini_batch_size, shuffle= True):
        val_loss.append(test_data(pad_batch(token).cuda(), Variable(torch.from_numpy(label), requires_grad= False).cuda(), 
                                  word_attn_model, sent_attn_model))
    return np.mean(val_loss)


# In[28]:

import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# ## Training

# In[31]:

def train_early_stopping(mini_batch_size, X_train, y_train, X_test, y_test, word_attn_model, sent_attn_model, 
                         word_attn_optimiser, sent_attn_optimiser, loss_criterion, num_epoch, 
                         print_val_loss_every = 1000, print_loss_every = 50):
    start = time.time()
    loss_full = []
    loss_epoch = []
    accuracy_epoch = []
    loss_smooth = []
    accuracy_full = []
    epoch_counter = 0
    g = gen_minibatch(X_train, y_train, mini_batch_size)
    for i in xrange(1, num_epoch + 1):
        try:
            tokens, labels = next(g)
            loss = train_data(tokens, labels, word_attn_model, sent_attn_model, word_attn_optimiser, sent_attn_optimiser, loss_criterion)
            acc = test_accuracy_mini_batch(tokens, labels, word_attn_model, sent_attn_model)
            accuracy_full.append(acc)
            accuracy_epoch.append(acc)
            loss_full.append(loss)
            loss_epoch.append(loss)
            # print loss every n passes
            if i % print_loss_every == 0:
                print 'Loss at %d minibatches, %d epoch,(%s) is %f' %(i, epoch_counter, timeSince(start), np.mean(loss_epoch))
                print 'Accuracy at %d minibatches is %f' % (i, np.mean(accuracy_epoch))
            # check validation loss every n passes
            if i % print_val_loss_every == 0:
                val_loss = check_val_loss(X_test, y_test, mini_batch_size, word_attn_model, sent_attn_model)
                print 'Average training loss at this epoch..minibatch..%d..is %f' % (i, np.mean(loss_epoch))
                print 'Validation loss after %d passes is %f' %(i, val_loss)
                if val_loss > np.mean(loss_full):
                    print 'Validation loss is higher than training loss at %d is %f , stopping training!' % (i, val_loss)
                    print 'Average training loss at %d is %f' % (i, np.mean(loss_full))
        except StopIteration:
            epoch_counter += 1
            print 'Reached %d epocs' % epoch_counter
            print 'i %d' % i
            g = gen_minibatch(X_train, y_train, mini_batch_size)
            loss_epoch = []
            accuracy_epoch = []
    return loss_full


# In[32]:
def train_main():
    word_attn = AttentionWordRNN(batch_size=64, num_tokens=81132, embed_size=300, 
                                 word_gru_hidden=100, bidirectional= True).cuda()

    sent_attn = AttentionSentRNN(batch_size=64, sent_gru_hidden=100, word_gru_hidden=100, 
                                 n_classes=10, bidirectional= True).cuda()

    learning_rate = 1e-1
    momentum = 0.9
    word_optmizer = torch.optim.SGD(word_attn.parameters(), lr=learning_rate, momentum= momentum)
    sent_optimizer = torch.optim.SGD(sent_attn.parameters(), lr=learning_rate, momentum= momentum)
    criterion = nn.NLLLoss()

    loss_full= train_early_stopping(64, X_train, y_train, X_test, y_test, word_attn, sent_attn, word_optmizer, sent_optimizer, 
                                criterion, 5000, 1000, 50)


    # In[34]:

    test_accuracy_full_batch(X_test, y_test, 64, word_attn, sent_attn)


    # In[35]:

    test_accuracy_full_batch(X_train, y_train, 64, word_attn, sent_attn)
'''



