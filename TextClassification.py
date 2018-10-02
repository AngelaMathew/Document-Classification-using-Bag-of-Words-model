#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_20newsgroups
import collections, re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd
import openpyxl
import nltk
import numpy as np
import string
from nltk import word_tokenize
import re
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        if re.match('[a-zA-Z]{3,}', item):
            stemmed.append(stemmer.stem(item))
    return stemmed
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [i for i in tokens if i not in string.punctuation]
    stems = stem_tokens(tokens, stemmer)
    return stems

categories = ['soc.religion.christian',
               'comp.graphics']
twenty_train = fetch_20newsgroups(subset='train',categories=categories, remove=('headers', 'footers', 'quotes'), shuffle=True)
vectorizer_s = CountVectorizer(min_df=3, analyzer='word',ngram_range=(1,1), stop_words="english", tokenizer=tokenize )
twenty_train_stemmed=vectorizer_s.fit_transform(twenty_train.data)
print("\n\ntfidf vectorisation docXterm matrix")
tfidf_transformer = TfidfTransformer()
twenty_train_stemmed_tfidf = tfidf_transformer.fit_transform(twenty_train_stemmed).toarray()
print(twenty_train_stemmed_tfidf.shape)
features=vectorizer_s.get_feature_names()
for t in features[:10]:
     print(t)

        
new_text = ['Divine jesus pray for us.','God is love','graphics and technology']
vectors_test = vectorizer_s.transform(new_text)

clf = MultinomialNB(alpha=.01)
clf.fit(twenty_train_stemmed_tfidf, twenty_train.target)
print(vectors_test.shape)

pred = clf.predict(vectors_test)
print(pred)
