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

import pyLDAvis
import pyLDAvis.gensim

from nltk import word_tokenize
from nltk.corpus import stopwords

STOPWORDS = stopwords.words('english')
STOPWORDS.extend(['amp','via','https','covid','covid-19','covid19','coronavirus'])
pbar = ProgressBar()

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.basicConfig(filename='lda_model.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


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

	train_df = tweets_df.sample(frac = .8, random_state = 0).sort_index()
	test_df = tweets_df[~tweets_df.isin(train_df)].dropna()

	#  clean training text and train lda model
	tweets_series = train_df['full_text']
	tokenized_data = []
	print('cleaning the data')
	for text in pbar(tweets_series):
	    tokenized_data.append(clean_text_pd(text))

	# bigrams
	print('making bigrams')
	bigram = gensim.models.Phrases(tokenized_data, min_count = 15)
	bigram_mod = models.phrases.Phraser(bigram)
	tokenized_data_bigram = [bigram_mod[review] for review in tokenized_data]

	print('making dictionary and corpus')
	dictionary = corpora.Dictionary(tokenized_data_bigram)
	dictionary.filter_extremes(no_below = 10, no_above = .3)
	dictionary.compactify()
	print('building the model!')
	corpus = [dictionary.doc2bow(text) for text in tokenized_data_bigram]
	lda_model = models.LdaMulticore(corpus = corpus, num_topics = num_topics, id2word = dictionary, 
		eval_every = None, iterations = 100)
	lda_model.save('lda_model.model')

	#pyLDAvis output
	print('creating the pyLDAvis html output. use this to make your topics look pretty and easy to understand')
	tweets_vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
	pyLDAvis.save_html(tweets_vis, 'lda_model.html')

	# determine topic make up for each tweet in the training set
	train_vecs = []
	#top_top = []
	print('running the model on each tweet in the training set')
	pbar2 = ProgressBar()
	for tweet in pbar2(tokenized_data_bigram):
	    train_vecs.append(lda_model.get_document_topics(tweet, minimum_probability=0.0))

	


	# move everything to folder!
	os.mkdir(output)
	for file in glob.glob('lda_model*'):
		shutil.move(file,output)

	print('Done! Nice work.')



main()