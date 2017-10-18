# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import unicodedata
from nltk import word_tokenize
import sys, re
import pandas as pd
import numpy as np
import json
import argparse
import json

'''
Change of this file. 
1. simplify the preprocessing process
2. generate one sentence everytime. 
3. remove the degree of lemmatize
4.
5. Down sampling the popular response. 

'''

#text = clean_str(text.strip()) if clean else text.strip()


def tokenize_url(instring):
    reg = re.compile(r'http[s]?:(//)?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', re.IGNORECASE)
    reg1=re.compile("([0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}|(((news|telnet|nttp|file|http|ftp|https)://)|(www|ftp)[-A-Za-z0-9]*\\.)[-A-Za-z0-9\\.]+)(:[0-9]*)?/[-A-Za-z0-9_\\$\\.\\+\\!\\*\\(\\),;:@&=\\?/~\\#\\%]*[^]'\\.}>\\),\\\"]")
    reg2=re.compile("([0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}|(((news|telnet|nttp|file|http|ftp|https)://)|(www|ftp)[-A-Za-z0-9]*\\.)[-A-Za-z0-9\\.]+)(:[0-9]*)?")
    #reg3=re.compile("(~/|/|\\./)([-A-Za-z0-9_\\$\\.\\+\\!\\*\\(\\),;:@&=\\?/~\\#\\%]|\\\\)+")
    reg4= re.compile("'\\<((mailto:)|)[-A-Za-z0-9\\.]+@[-A-Za-z0-9\\.]+")
    instring = re.sub(reg, '_url_', instring)
    instring = re.sub(reg1, '_url_', instring)
    instring = re.sub(reg2, '_url_', instring)
    #instring = re.sub(reg3, '_url_', instring)
    #instring = re.sub(reg4, '_url_', instring)
    return instring


def tokenize_email(instring):
    reg = re.compile (r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
    return re.sub(reg, '_email_', instring)

def tokenize_date(instring):
    reg = re.compile(r'[0-9]+\/[0-9]+(\/\*\*\*\*|\/[0-9]+)?', re.IGNORECASE)
    return re.sub(reg, '_date_', instring)

def tokenize_cost(instring):
    outstring = instring
    reg = re.compile(r'\$?[0-9]+(\.[0-9]+)?\/mo(nth)?', re.IGNORECASE)
    outstring = re.sub(reg, '_cost_', outstring)
    reg1 = re.compile(r'\$[0-9]+(\.[0-9]+)?')
    return re.sub(reg1, '_cost_', outstring)

def tokenize_ispeed(instring):
    reg = re.compile(r'[0-9]+(\.[0-9]+)? ?(mb(\/)?s|[km]bps)', re.IGNORECASE)
    return re.sub(reg, '_ispeed_', instring)

def tokenize_phonenum(instring):
    reg = re.compile (r'([0-9*]{3}[-\.\s]??[0-9*]{3}[-\.\s]??[0-9*]{4}|[0-9*]{1,2}[-\.\s]??[0-9*]{3}[-\.\s]??[0-9*]{3}[-\.\s]??[0-9*]{4})')
    instring = re.sub(reg, '_phone_', instring)
    reg1 = re.compile (r'([a-zA-Z0-9*]{3}[-][a-zA-Z0-9*]{3}[-][a-zA-Z0-9*]{4}|[0-9*]{1,2}[-][a-zA-Z0-9*]{3}[-][a-zA-Z0-9*]{3}[-][a-zA-Z0-9*]{4})')
    instring = re.sub(reg1, '_phone_', instring)
    reg2 = re.compile (r'(\([a-zA-Z0-9*]{3}\)\s*[a-zA-Z0-9*]{3}[-\.\s]??[a-zA-Z0-9*]{4}|[a-zA-Z0-9*]{3}[-\.\s]??[a-zA-Z0-9*]{4})')
    #instring = re.sub(reg1, '_phone_', instring)
    #reg = re.compile (r'[a-zA-Z0-9*]{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
    #reg = re.compile (r'(XXXXXX[0-9]{4})|(\\(XXXX\\)[0-9]{3}-[0-9]{4})|(XXXX [0-9]{3} [0-9]{4})')
    return instring

rgx_DataVolume = re.compile ('[0-9]*gb', re.IGNORECASE)
rgx_Percentage = re.compile ('[0-9]+\.[0-9]*%')
rgx_Day = re.compile ('[0-9]?[0-9](st|nd|rd|th)', re.IGNORECASE)
#rgx_Month = re.compile ('(january)|(jan)|(february)|(feb)|(march)|(mar)|(april)|(apr)|(may)|(june)|(jun)|(july)|(jul)|(august)|(aug)|(september)|(sep)|(october)|(oct)|(november)|(nov)|(december)|(dec)', re.IGNORECASE)
rgx_Year = re.compile ('(19[0-9]{2})|(20[0-9]{2})')
rgx_Num = re.compile(r'[0-9]+(\.[0-9]+)?')
rgx_Accountnum = re.compile(r'[0-9*]{13,18}')
rgx_Time = re.compile(r'([0-9]|0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]')

# -----------------------------------------------------
def preclean_text (_text):
    # nltk.word_tokenize doesn't seem to handle these correctly. ??
    # Tokenization/string cleaning
    _text = _text.replace ("I'm ", "I am ")
    _text = re.sub(r"\'s", " \'s", _text)
    _text = re.sub(r"\'ve", " \'ve", _text)
    _text = re.sub(r"n\'t", " n\'t", _text)
    _text = re.sub(r"\'re", " \'e", _text)
    _text = re.sub(r"\'d", " \'d", _text)
    _text = re.sub(r"\'ll", " \'ll", _text)
    _text = re.sub(r"\s{2,}", " ", _text)
    _text = re.sub(r"[-]{8,}", "--", _text)
    _text = tokenize_url(_text)
    _text = tokenize_email(_text)
    _text = tokenize_date(_text)
    _text = tokenize_cost(_text)
    _text = tokenize_ispeed(_text)
    _text = re.sub(rgx_Accountnum, "_account_", _text)
    _text = tokenize_phonenum(_text)
    _text = re.sub(rgx_DataVolume, "_data_", _text)
    _text = re.sub(rgx_Time, "_time_", _text)
    _text = re.sub(rgx_Percentage, "_percentage_", _text)
    _text = re.sub(rgx_Day, "_day_", _text)
    #_text = re.sub(rgx_Month, "_month_", _text)
    _text = re.sub(rgx_Year, "_year_", _text)
    _text = re.sub(rgx_Num, "_num_", _text)

    _text = re.sub(r",", " , ", _text)
    _text = re.sub(r"!", " ! ", _text)
    _text = re.sub(r"\(", " \( ", _text)
    _text = re.sub(r"\)", " \) ", _text)
    _text = re.sub(r"\?", " \? ", _text)
    
    _text = _text.replace ('.', ' . ')
    #_text = _text.replace (',', ' , ')
    _text = _text.replace (':', ' : ')
    _text = _text.replace (';', ' ; ')
    #_text = _text.replace ('?', ' ? ')
    #_text = _text.replace ('!', ' ! ')
    #_text = _text.replace ('(', ' ( ')
    #_text = _text.replace (')', ' ) ')
    _text = _text.replace ('"', ' " ')
    _text = _text.replace ('[', ' [ ')
    _text = _text.replace (']', ' ] ')
    _text = _text.replace ('{', ' { ')
    _text = _text.replace ('}', ' } ')
    _text = _text.replace ('-', ' - ')
    _text = _text.replace ('=', ' = ')
    _text = _text.replace ('+', ' + ')
    _text = _text.replace ('*', ' * ')
    _text = _text.replace ('~', ' ~ ')
    _text = _text.replace ('|', ' | ')
    _text = _text.replace ('#', ' # ')
    _text = _text.replace ('\\n', ' ')
    _text = _text.replace ('\\', ' ')
    _text = _text.replace ('…', ' ')
    _text = _text.replace ('“', ' ')
    _text = _text.replace ('”', ' ')
    _text = _text.replace ('，', ' , ')
    #_text = _text.replace ('_', ' _ ')
    _text = _text.replace ('#', ' # ')
    '''
    _text = _text.replace ('’', ' \' ')
    _text = _text.replace ('\'', ' \' ')
    '''
    _text = re.sub(r"\s{2,}", " ", _text)

    return _text.strip()

# -----------------------------------------------------          
def load_conversations(csv_file='/D/data/autosuggest_data/cc/cc_20170204/'):
    df = pd.read_csv(csv_file)
    print('Finish reading csv file: {}.'.format(csv_file))
    #df = df[["RowKey","eventflagfromrep","text"]]
    df = df[[not x for x in df['isautogenerated']]]
    df = df[["rowkey","eventflagfromrep","text"]]
    df.columns = ["conversationid","eventflagfromrep","text"]
    df = df[~pd.isnull(df.text)]
    return df


def conversation_save(data1, file, args,  MAX_wps = 50, MAX_turn =50, saving_starts_turnn=6 ):
    conversation_begin_symbol = " __SOC__ "
    customer_begin_symbol = "<cus__ "
    customer_end_symbol = " __cus>"
    agent_begin_symbol = "<agent__ "
    agent_end_symbol = " __agent>"

    indicator = file.split('_')[0]
    #fileout = args.dir+indicator+'_v'+args.version
    #f_tgt = open(args.outdir +'/'+ 'tgt-'+indicator+'_v'+args.version+'.txt', 'w')
    #f_src = open(args.outdir +'/'+ 'src-'+indicator+'_v'+args.version+'.txt', 'w')

    conv_n = 0
    pairn = 0
    context_stats = []
    Autt_stats = []

    utter_n = 0
    old_id = -1 #data1.iloc[0]['conversationid']
    last_speaker = data1.iloc[0]['eventflagfromrep']

    #a_utt = ''
    #c_utt = ''
    context = []
    conversation = []
    replies = []
    all_turn = []
    turns = []
    speaker = []
    all_speaker = []

    #with open('prob_dict.json', 'r') as f:
    #    prob_dict = json.load(f)
    #print(prob_dict)

    for i in range(len(data1)):
        #print(i)
        new_id = data1.iloc[i]['conversationid']
        this_speaker = data1.iloc[i]['eventflagfromrep']
        text = data1.iloc[i]['text']
        utt = preclean_text(text.lower())
        if len(utt.split(' '))> MAX_wps:
            utt = ' '.join(utt.split(' ')[-MAX_wps:])

        if new_id != old_id:
            conv_n += 1
            utter_n = 0
            context=[]
            turns = []
            speaker = []
            #context = conversation_begin_symbol
            context.append(conversation_begin_symbol)
            turns.append(utter_n)
            speaker.append(this_speaker)

        # customer is speaking 
        if this_speaker == False:
            c_utt = customer_begin_symbol + utt + customer_end_symbol
            #context = context + ' ' + c_utt
            context.append(c_utt)
            turns.append(utter_n)
            speaker.append(this_speaker)

        if this_speaker == True:
            a_utt = agent_begin_symbol + utt + agent_end_symbol

            if utter_n >= saving_starts_turnn:
                '''
                #'save the status'
                if a_utt in prob_dict:
                    #print(a_utt)
                    p = prob_dict[a_utt]
                    if p<1 and np.random.binomial(1, p, 1) == 0:
                        #print(a_utt)
                        continue
                '''
                pairn +=1


                Autt_stats.append(len(a_utt.split(' ')))
                context_arr = [w for sent in context for w in sent.split(' ') ]
                context_stats.append(len(context_arr))
                if len(context) > MAX_turn:
                    context = context[-MAX_turn:]

                conversation.append(context)
                all_turn.append(turns)
                all_speaker.append(speaker)
                replies.append(a_utt)

            context.append(a_utt)
            turns.append(utter_n)
            speaker.append(this_speaker)

        utter_n += 1
        old_id = new_id
        last_speaker = this_speaker

    filename=args.outdir +'/'+ 'src-'+indicator+'_v'+args.version+'.txt'
    data = {'context':conversation, 'replies':replies, 'speaker':all_speaker, 'conv_turns':all_turn}
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)

    print('In total {} conversations'.format(conv_n))
    print('Built {} seq-to-seq pairs'.format(pairn))
    print('==simple data stats: ==')
    print('Context: ')
    hist, bin_edges = np.histogram(context_stats, bins=5)
    print('length ranges of: ')
    print(bin_edges)
    print('Counts of context: ')
    print(hist)
    print('Replies: ')
    hist, bin_edges = np.histogram(Autt_stats, bins=5)
    print('length ranges of: ')
    print(bin_edges)
    print('Counts of replies: ')
    print(hist)
    print('File saved to: {} and {}.'.format(args.outdir +'/'+ 'tgt-'+indicator+'_v'+args.version+'.txt',args.outdir +'/'+ 'src-'+indicator+'_v'+args.version+'.txt' ))
    print('\n')
    #return pairs


# -----------------------------------------------------        
def pairs_tofile(pairs, indicator):
    f_tgt = open('/awsnas/data/convdata/tgt-'+indicator+'.txt', 'w')
    f_src = open('/awsnas/data/convdata/src-'+indicator+'.txt', 'w')
    for p in pairs:
        f_tgt.write(p[0])
        f_tgt.write('\n')
        f_src.write(p[1])
        f_src.write('\n')
    f_tgt.close()
    f_src.close()


def main ():
    parser = argparse.ArgumentParser(description='process from raw data for seq2seq training')
    parser.add_argument('-indir', default='/awsnas/data/dialogue_csv', type=str, help='location of the file, e.g awsnas')
    parser.add_argument('-outdir', default='/awsnas/data/convdata', type=str, help='location of the file, e.g awsnas')
    parser.add_argument('--files', default=[], nargs='+', type=str, help='name of files to process')
    parser.add_argument('--version', default='', type=str, help='version of the file')
    args = parser.parse_args()
   
    print(args.files)

    for file in args.files:
        filein = args.indir+'/'+str(file)

        #indicator = file.split('_')[0]
        #fileout = args.dir+indicator+'_v'+args.version

        dff= load_conversations(filein)
        conversation_save(dff, file, args)
        

if __name__ == '__main__':
    sys.exit (main ())

