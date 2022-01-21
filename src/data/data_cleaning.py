import nltk
import data_reader
import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from copy import deepcopy


span_tokenize_dict = {}


#def __regex_separator(text,separator):
#   # separator =["–"]#["–"]
#    for sep in separator: 
#       text= text.replace(sep," ")
#    return text


#def separate_words(df,separator=["-"]):
#    columns=["passage","answer","question"]
#    for col in columns:
#        df[col] = df.apply(lambda x: __regex_separator(x[col],separator), axis = 1)
#    return df



def get_answer_start_end(passage,answer_text,answer_start):
    answer_end = len(answer_text) + answer_start
    
    if passage not in span_tokenize_dict.keys():
        span_tokenize_dict[passage] = span_tokenize(passage)


    
    interval = [i for i, (s, e) in enumerate(span_tokenize_dict[passage]) if e >= answer_start and s <= answer_end]
    if len(interval) <1:
       #raise Exception(interval + " is empty.") 
       mamma= [answer_text]#[str(passage)[96]]
       print(mamma)
       
       return mamma
       # 
       #return [-1,-1]
    return [min(interval),max(interval)]


def add_labels(df):
    df["label"] = df.apply(lambda x: get_answer_start_end(x["passage"], x["answer"],x["answer_start"]), axis = 1)
    return df


def data_cleaning(df):
    #df = separate_words(df)
    
    return add_labels(df).drop(axis=1, columns='answer_start')
  




def tokenizers():
    tokenizer1 = RegexpTokenizer(r'\d[.,]\d+|\w+|\S')
    tokenizer2 = RegexpTokenizer(r'\d[.,]\d+|\w+|\S|.')
    return tokenizer1, tokenizer2


def group_tokens(t,t_with_spaces):
    t1=deepcopy(t)
    first_item_found = False
    first_string = ""
    j = 0
    for el in t_with_spaces:
        correspondence_not_found = False
        if j < len(t1):
            if t1[j] == el:
                if not first_item_found:
                    t1[j] = first_string + t1[j]
                    first_item_found = True
                j += 1
            else:
                correspondence_not_found = True
        else:
            correspondence_not_found = True

        if correspondence_not_found:
            if first_item_found:
                t1[j-1] = t1[j-1] + el
            else:
                first_string += el

    return t1



def tokenize_with_spaces(sentence):
    t1,t2 = tokenizers()
    sentence_tokenized = t1.tokenize(sentence)
    sentence_tokenized_with_spaces = t2.tokenize(sentence)
    t_grouped = group_tokens(sentence_tokenized, sentence_tokenized_with_spaces)
    return t_grouped


def span_tokenize(sentence):
    tokenized_sentence = tokenize_with_spaces(sentence)
    span_list = []
    j = 0
    for el in tokenized_sentence:
        span_list.append((j, j + len(el) -1))
        j += len(el)
    return span_list





#s= "   anna.    va  "
#   0123456789012345
#print(tokenize_with_spaces(s))
#print(span_tokenize(s))