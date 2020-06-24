#!/usr/bin/env python3

import subprocess
import pandas as pd 
import GetOldTweets3 as got 

args = ['GetOldTweets3','--usernames-from-file', 'ITS/users_list_got.csv','--output','ITS/got_tweets.csv']
t = subprocess.Popen(args)
t.wait()

print('reading in files...')
selected_users = pd.read_csv('ITS/selected_users_itsa.csv')
got_tweets = pd.read_csv('ITS/got_tweets.csv')

print('merging data frames...')
got_tweets = got_tweets.drop(columns = ['permalink','geo','mentions','hashtags','to'])
got_tweets = got_tweets.rename(columns = {'username':'screen_name'})
selected_users_full = got_tweets.merge(selected_users, on = 'screen_name')

print('saving top users...')
selected_users_full.to_csv('ITS/selected_users_full.csv.gz')

