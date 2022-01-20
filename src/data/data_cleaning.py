import nltk
import data_reader
import pandas as pd
import numpy as np
from nltk.tokenize.util import string_span_tokenize


def get_tokenizer():
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    # or tokenizer = nltk.tokenize.WhitespaceTokenizer()
    return tokenizer


def __regex_separator(text):
    separator = ["-"]
    for sep in separator: 
       text= text.replace(sep," "+sep+" ")
    return text


def separate_words(df):
    columns=["passage","answer","question"]
    for col in columns:
        df[col] = df.apply(lambda x: __regex_separator(x[col]), axis = 1)
    return df


def get_answer_start_end(passage,answer_text,answer_start):
    answer_end = len(answer_text) + answer_start
    tokenizer = get_tokenizer()

    
    interval = [i for i, (s, e) in enumerate(tokenizer.span_tokenize(passage)) if s >= answer_start and e <= answer_end]
    if len(interval) <1:
        #raise Exception(interval + " is empty.") 
        return [-1,-1]
    return [min(interval),max(interval)]


def add_labels(df):
    df["label"] = df.apply(lambda x: get_answer_start_end(x["passage"], x["answer"],x["answer_start"]), axis = 1)
    return df


def data_cleaning(df):
    #df = separate_words(df)
    return add_labels(df).drop(axis=1, columns='answer_start')
    

#passage = 'the bee buzzed loudly'
#answer_text ="bee buzzed"
#answer_start = 4  # substring begin and end

#print(get_answer_start_end(passage,answer_text,answer_start))

#df= pd.DataFrame(np.array([["ciao giovanni come-stai","ciao giovanni come-stai","ciao giovanni come-stai"],
#                           ["ciao giovanni come-stai","ciao giovanni come-stai","ciao giovanni come-stai"],
#                           ["ciao giovanni come-stai","ciao giovanni come-stai","ciao giovanni come-stai"]]),columns=["passage","question","answer"])
#print("sono qui",df[0:1])
#print(separate_words(df)[0:1])








