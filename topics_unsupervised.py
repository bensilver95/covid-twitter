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


# what i hope this script can eventually do:
# you input a dataset and a number of topics
# it gets you an lda model, and ldaseq model, some sort of pyldavis visualization, and a graph of topic frequency in test data

# tex cleaning function
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

	#  clean text and train lda model
	tweets_series = tweets_df['full_text']
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

	# determine topic make up for each tweet
	tweet_topics = []
	top_top = []
	print('running the model on each tweet')
	pbar2 = ProgressBar()
	for tweet in pbar2(corpus):
		test = lda_model.get_document_topics(tweet, minimum_probability=0.0)
	    tweet_topics.append(test)
	    tweet_topics_df = pd.DataFrame(test, columns = ['topic num','topic pct'])
	    top_top.append(tweet_topics_df['topic num'][tweet_topics_df.idxmax(axis = 0)[1]] + 1)
	tweets_df['top_topic'] = top_top
	tweets_df['topic_vector'] = tweet_topics
	tweets_df.to_csv('lda_model_tweets_topics.csv.gz', index = False)

	# gather top topic data into format that can be graphed in R
	print('gather data topics into format for graphing')
	dates = tweets_df['date'].unique()
	graph = np.zeros((len(dates),num_topics + 1))
	graph = pd.DataFrame(graph)
	for date in range(len(dates)):
	    graph.iloc[date,0] = dates[date]
	    date_df = tweets_df[tweets_df['date'] == dates[date]]
	    for topic in range(num_topics):
	        graph.iloc[date,topic+1] = len(date_df[date_df['top_topic'] == topic])
	graph.to_csv('lda_model_topic_graph.csv', index = False)


	# run ldaseq
#	print('now use the data to create a dynamic topic model')
#	timeslice = tweets_df.groupby(by = 'date').count()['id'].to_list()
#	ldamodel_seq = models.ldaseqmodel.LdaSeqModel(corpus=corpus, id2word=dictionary, time_slice=timeslice, num_topics=num_topics)
#	ldamodel_seq.save('lda_model_dynamic.model')

	# move everything to folder!
	os.mkdir(output)
	for file in glob.glob('lda_model*'):
		shutil.move(file,output)

	print('Done! Nice work.')



main()
