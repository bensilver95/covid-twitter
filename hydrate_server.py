#!/usr/bin/env python3

#
# This script will walk through all the tweet id files and
# hydrate them with twarc. The line oriented JSON files will
# be placed right next to each tweet id file.
#
# Note: you will need to install twarc, tqdm, and run twarc configure
# from the command line to tell it your Twitter API keys.
#

import gzip
import json
import os.path
import sys

from tqdm import tqdm
from twarc import Twarc
from pathlib import Path
import pandas as pd

twarc = Twarc()
data = 'full_dataset-clean.tsv.gz'
data_df = pd.read_csv(data, sep = '\t')
dates = data_df['date'].unique()
if len(sys.argv) > 1:
    start_date = str(sys.argv[1])
    dates = dates[dates.tolist().index(start_date) + 1:]
if len(sys.argv) > 2:
    end_date = str(sys.argv[2])
    dates = dates[:dates.tolist().index(end_date) + 1]


def main():
    for date in dates:
        data_df_date = data_df[data_df['date'] == date]
        data_df_date = data_df_date.reset_index()
        hydrate(data_df_date)


def _reader_generator(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)


def raw_newline_count(fname):
    """
    Counts number of lines in file
    """
    file = pd.read_csv(fname, sep = '\t')
    return len(file)


def hydrate(id_file):
    print('hydrating {}'.format(id_file))

    gzip_path = 'jsons/' + id_file['date'][0].replace('/','_') + '.jsonl.gz'
    if os.path.isfile(gzip_path):
        print('skipping json file already exists: {}'.format(gzip_path))
        return

    num_ids = len(id_file)

    with gzip.open(gzip_path, 'w') as output:
        with tqdm(total=num_ids) as pbar:
            for tweet in twarc.hydrate(id_file['tweet_id']):
                if tweet['lang'] == 'en':
                    output.write(json.dumps(tweet).encode('utf8') + b"\n")
                    pbar.update(1)


if __name__ == "__main__":
    main()
