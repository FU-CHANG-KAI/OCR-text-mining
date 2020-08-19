from collections import Counter
from scrapt import scrapt_html_to_df
import os
import string
import re

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as bs
import pickle

import spacy

from nltk.corpus import stopwords

from spacy.lemmatizer import Lemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 

import pickle_file 

en = spacy.load('en_core_web_sm')
en.max_length = 3000000
#lemmatize = en.Defaults.create_lemmatizer()

tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer() 

START_SPEC_CHARS = re.compile('^[{}]+'.format(re.escape(string.punctuation)))
END_SPEC_CHARS = re.compile('[{}]+$'.format(re.escape(string.punctuation)))



def _get_stop_words(strip_handles, strip_rt):
    ''' Returns stopwords '''
    stop_words = (stopwords.words('english'))
    if strip_rt: stop_words += ['rt']
    return set(stop_words)

def _has_digits(token):
    ''' Returns true if the given string contains any digits '''
    return any(char.isdigit() for char in token)

class Dataset():
    def __init__(self, mode, strip_handles=True, 
                             strip_rt=True, 
                             strip_digits=True,
                             strip_hashtags=False,
                             test_size=0.3):
        # Get raw data
        self.corpus, self.y = self._get_training_data()

        # 10 April: Preprocess 'contents' using nltk and spacy and get back to the DataFrame

        if mode == "tf_idf":
            self.X = self._tokenize_with_lemma_tfidf(self.corpus, 
                                            strip_handles, 
                                            strip_rt,
                                            strip_digits)
        elif mode == "doc2vec":
            self.X = self._tokenize_with_lemma_d2v(self.corpus, 
                                            strip_handles, 
                                            strip_rt,
                                            strip_digits)

    def _get_training_data(self):
        df = scrapt_html_to_df
        X = df['contents'].tolist()
        y = df['book_name'].tolist()
        print("Complete to get_training_data")
        return X, y

    def _tokenize_with_lemma(self, corpus, strip_handles=True, strip_rt=True, strip_digits=True):
        ''' Tokenize and lemmatize using Spacy '''
        
        stop_words = _get_stop_words(strip_handles, strip_rt)

        output = []
        for text in corpus:
            # Tokenize the document.
            tokens = tokenizer.tokenize(text.lower())

            #tokens = [lemmatizer.lemmatize(token) for token in tokens]
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
            
            #tokens = [stemmer.stem(token) for token in tokens]

            # Remove punctuation tokens.
            tokens = [token for token in tokens if token not in string.punctuation+'…’']

            # Remove tokens wich contain any number.
            if strip_digits:
                tokens = [token for token in tokens if not _has_digits(token)]

            # Remove tokens without text.
            tokens = [token for token in tokens if bool(token.strip())]

            # Remove punctuation from start of tokens.
            tokens = [re.sub(START_SPEC_CHARS, '', token) for token in tokens]

            # Remove punctuation from end of tokens.
            tokens = [re.sub(END_SPEC_CHARS, '', token) for token in tokens]

            # Remove stopwords from the tokens
            tokens = [token for token in tokens if token not in stop_words]

            output.append(tokens)

        return output

if __name__ == '__main__':
    ds = Dataset("doc2vec")  

    return ds.X
    pickle_file.save(ds.X)
