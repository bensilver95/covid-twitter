#!/usr/bin/env python3

import gc
import sys
import time
import json
import gzip
import glob
import tqdm
import urllib
import os.path
import geocoder
import requests
import progressbar
import numpy as np
import pandas as pd

# define a function for bot pruning
def filetime_conv(created_at):
    outputdir = 'jsons/'
    file_for_conv = outputdir + str(created_at).split(' ')[0] + '.jsonl.gz'
    timect = pd.to_datetime(time.ctime(os.path.getctime(file_for_conv)))
    return timect

def url_filter(url):
    if not url:
        return 1
    elif url[0]['display_url'].startswith('twitter.com'):
        return 1
    else:
        return 0

# function for converting coordinates to county fips
def coord_to_fips(x):
    params = urllib.parse.urlencode({'latitude': x[0], 'longitude':x[1], 'format':'json'})
    url = 'https://geo.fcc.gov/api/census/block/find?' + params
    response = requests.get(url)
    data = response.json()
    return data['County']['FIPS']

filedir = 'jsons/'
filelist = sorted(glob.glob(filedir + '*gz'))
if len(sys.argv) > 1:
    start_date = str(sys.argv[1])
    start_file = start_date + '.jsonl.gz'
    filelist = filelist[filelist.index(filedir + start_file) + 1:]
if len(sys.argv) > 2:
    end_date = str(sys.argv[2])
    end_file = end_date + '.jsonl.gz'
    filelist = filelist[:filelist.index(filedir + end_file) + 1]



#tweets = pd.DataFrame(columns = ['created_at','id','full_text','retweet_count','favorite_count','entities',
#    'screen_name','user_id','location','followers_count','user_created_at','statuses_count'])
keys = ['created_at','id','full_text','retweet_count','favorite_count','entities','user']
user_keys = ['screen_name','id','location','followers_count','created_at','statuses_count']
df_iters = []
tweets_pruned = pd.DataFrame(columns = ['created_at','id','full_text','retweet_count','favorite_count','entities',
                                 'screen_name','user_id','location','followers_count','user_created_at','statuses_count'])

    
# pull relevant information, save to df
pbar = progressbar.ProgressBar()
for file in pbar(filelist):
    j = pd.read_json(file, lines = True, chunksize = 100000)
    df_iters.append(j)


for i in range(len(df_iters)):
    df = df_iters[i]
    file = filelist[i]
    json_name = file.split('jsons/')[1]
    print('analyzing tweets from ' + json_name)
    for chunk in df:
        tweets = pd.DataFrame(chunk,columns = keys).reset_index(drop = True)
        tweets_user = pd.DataFrame(chunk['user'].to_list(), columns = user_keys)
        tweets_user = tweets_user.rename(columns = {'id':'user_id','created_at':'user_created_at'})
        tweets = pd.concat([tweets,tweets_user], axis = 1)
        tweets = tweets.drop(columns = 'user')
         
        tweets = tweets.drop_duplicates(subset = 'full_text', keep = False) # remove duplicates
        tweets = tweets[tweets['full_text'].str.contains('coronavirus|covid|corona virus', case = False)] #only coronavirus, corona virus or covid
        tweets = tweets[~tweets['favorite_count'].isna()].reset_index(drop = True) # get rid of junk

        # prune bots!
        print('pruning bots')
        print('based on tweet rate')
        tqdm.tqdm.pandas()
        hydration_date = tweets['created_at'].progress_apply(filetime_conv) # this could be sped up I think
        tweets['user_created_at'] = pd.to_datetime(tweets['user_created_at'])
        tweets['daily_tweet_rate'] = tweets['statuses_count']/(hydration_date.dt.date - tweets['user_created_at'].dt.date).dt.days
        tweets = tweets[tweets['daily_tweet_rate'] < 100]. reset_index(drop = True)
        print('get rid of tweets with external link')
        entities = pd.DataFrame(tweets['entities'].to_list())
        tqdm.tqdm.pandas()
        tweets = tweets[entities['urls'].progress_apply(url_filter) == 1]

        tweets_pruned = tweets_pruned.append(tweets, ignore_index = True)


#print('saving ' + str(len(tweets)) + ' tweets to file')
#tweets.to_csv('big_csv_files/all_tweets.csv.gz', index = False)

# Only save tweets where the user has a location listed
print('filtering out users with no location set')
tweets_location = tweets_pruned[tweets_pruned['location'] != ''].reset_index(drop = True)
#print('saving ' + str(len(tweets_location)) + ' tweets to file')
#tweets_location.to_csv('big_csv_files/location_tweets.csv.gz', index = False)

# Only save tweets where the location is in the USA
places_nocase = pd.read_excel('us_places.xlsx',sheet_name = 'no_case')['places'].to_list()
places_case = pd.read_excel('us_places.xlsx',sheet_name = 'case')['places'].to_list()
places = pd.read_excel('us_places.xlsx', sheet_name = 'all')

print('manually filtering out users outside of the US so geocoding works better')
tweets_usa = tweets_location[tweets_location['location'].str.contains('|'.join(places_nocase),case = False)|tweets_location['location'].str.contains('|'.join(places_case))].reset_index(drop = True)
tweets_usa['location'] = tweets_usa['location'].str.replace(', CA',', California',case = True) # california fix for geocoder

state = tweets_usa['id'].where(tweets_usa['id'] == 1)
print('assigning states...')
pbar = progressbar.ProgressBar()
for i in pbar(range(len(places))):
    if len(places.iloc[i,0]) == 2:
        temp = tweets_usa['location'].where(tweets_usa['location'].str.contains(places.iloc[i,0]))
    else:
        temp = tweets_usa['location'].where(tweets_usa['location'].str.contains(places.iloc[i,0], case = False))
    temp[~temp.isna()] = places.iloc[i,1]
    state = state.combine_first(temp)
tweets_usa['state'] = state
tweets_usa = tweets_usa[~tweets_usa['state'].isna()].reset_index(drop = True)

if len(sys.argv) > 3:
    print('saving ' + str(len(tweets_usa)) + ' tweets to a test file')
    tweets_usa.to_csv('big_csv_files/usa_tweets.csv.gz')
else:
    print('adding ' + str(len(tweets_usa)) + ' tweets to file')
    tweets_usa_old = pd.read_csv('big_csv_files/usa_tweets.csv.gz')
    tweets_usa = tweets_usa_old.append(tweets_usa, ignore_index = True)
    tweets_usa = tweets_usa.drop_duplicates(subset = 'full_text', keep = False) # remove duplicates
    print('now have ' + str(len(tweets_usa)) + ' total tweets')
    tweets_usa.to_csv('big_csv_files/usa_tweets.csv.gz', index = False)

print('extracting top users...')
tweets_unique_users = tweets_usa.drop_duplicates(subset = 'screen_name').reset_index(drop = True)
selected_users = tweets_unique_users['screen_name'].sample(n = 5000, random_state = 0).to_list()
tweets_selected_users = tweets_unique_users[tweets_unique_users['screen_name'].isin(selected_users)].reset_index(drop = True)
selected_users_df = tweets_selected_users[['screen_name','user_id','location','followers_count','statuses_count','daily_tweet_rate']]
users_list_got = selected_users_df[['screen_name']]
selected_users_df.to_csv('ITS/selected_users_itsa.csv', index = False)
users_list_got.to_csv('big_csv_files/users_list_got.csv',index = False, header = False)
