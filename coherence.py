#!/usr/bin/env python3

import re
import os
import sys
import glob
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from progressbar import ProgressBar

import gensim
from gensim import models, corpora

from nltk import word_tokenize
from nltk.corpus import stopwords

STOPWORDS = stopwords.words('english')
STOPWORDS.append('amp')
STOPWORDS.append('via')
STOPWORDS.append('https')
STOPWORDS.append('covid')
STOPWORDS.append('covid-19')
STOPWORDS.append('covid19')
STOPWORDS.append('coronavirus')
pbar = ProgressBar()

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.basicConfig(filename='lda_model.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# find optimal number of topics!
# text cleaning function
def clean_text_pd(text):
    tokenized_text = word_tokenize(text.lower())
    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    return cleaned_text

def main():

	num_topics = int(sys.argv[1])
	output = str(sys.argv[2])
	print('lets do some topic modeling! ' + str(num_topics) + ' topics are coming your way.')

	# load data
	print('reading the covid tweets file...')
	tweets_df = pd.read_csv('/home/bsilver/covid/big_csv_files/usa_tweets.csv.gz').drop(columns = ['Unnamed: 0'])
	created_dt = pd.to_datetime(tweets_df['created_at'])
	tweets_df['created_at'] = created_dt
	tweets_df['date'] = tweets_df['created_at'].dt.date

	#  train lda model
	tweets_series = tweets_df['full_text']
	tokenized_data = []
	print('cleaning the data')
	for text in pbar(tweets_series):
	    tokenized_data.append(clean_text_pd(text)) 
	dictionary = corpora.Dictionary(tokenized_data)
	dictionary.filter_extremes(no_below = 5, no_above = .3)
	dictionary.compactify()
	print('building the model!')
	corpus = [dictionary.doc2bow(text) for text in tokenized_data]

	coherence_values = []
	for i in range(5,25):
	    num_topics = i
	    print('*** building model for ' + str(num_topics) + ' topics ***')
	    lda_model_for_coherence = models.LdaMulticore(corpus = corpus, num_topics = num_topics, id2word = dictionary, 
	    	eval_every = None, iterations = 100)	    
	    coherence_model_lda = models.coherencemodel.CoherenceModel(model=lda_model, texts=tokenized_data, dictionary=dictionary, coherence='c_v')
	    coherence_values.append(coherence_model_lda.get_coherence())

	plt.plot(range(5,25),coherence_values)
	plt.xlabel('Number of topics')
	plt.ylabel('Coherence')
	plt.savefig('coherence_graph.jpg', quality = 95, dpi = 150)
