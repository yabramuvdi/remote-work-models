#%%

import pyreadr
import time
import os
import string
import pandas as pd
import numpy as np
import swifter
import multiprocessing
import gc
import psutil
from itertools import repeat
from tqdm.contrib.concurrent import process_map
import re
import random
import yaml
import subprocess
import argparse

# import our own module
import dict_methods_functions as dm

# main paths
input_path = "../input/"
output_path = "../output/"
config_path = "../config/"

# load parameters
with open(config_path + "config.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

#%%

#======
# Global parameters
#======

# define punctuation symbols to remove
punctuation = string.punctuation
punctuation = punctuation.replace("-", "")
punctuation = punctuation.replace("'", "")

# define tokenization pattern
tokenization_pattern = r'''
                        (?x)                # set flag to allow verbose regexps (to separate logical sections of pattern and add comments)
                        \w+(?:-\w+)*        # word characters with internal hyphens
                        \w+(?:'\w+)*        # word characters with internal apostrophes
                        |\b\w+\b            # single letter words
                        '''

#======
# 0. Load dictionaries and create regular expressions
#======

# load dictionary
dict_df = pd.read_csv(input_path + "dictionary.csv")
dict_list = dict_df["term"].tolist()

# generate a complete regular expression from the dictionary
dict_regex = dm.gen_regex_from_dict(dict_list)
dict_regex = re.compile(dict_regex, re.IGNORECASE)
print(f"Dictionary regex: {dict_regex}")

# load negation terms
df_neg = pd.read_csv(input_path + "negation.csv")

# load generic terms
df_generic = pd.read_csv(input_path + "generic.csv")

# generate a regular expression for generic terms
generic_regex  = ""
for _, row in df_generic.iterrows():
    term = row["term"]
    part = row["part of word"]

    if part == "no":
        # add word boundaries to the regex
        term = r"\b" + term + r"\b"
        generic_regex += term + r"|"
    elif part == "yes":
        # add only starting word boundary to the regex
        term = r"\b" + term
        generic_regex += term + r"|"

# clean last pipe and compile regex
generic_regex = generic_regex[:-1]
generic_regex = re.compile(generic_regex, re.IGNORECASE)

#%%

#=====
# 1. Define main functions
#=====

def read_r(file_path):
    result = pyreadr.read_r(file_path)
    df_temp = result[None].copy()
    # return only selected columns
    return df_temp[["seq_id", "sequence"]]


def partition_df(df, col_text="sequence", join_char=" "):

    print(f"Number of sequences in file: {len(df)}")

    #=====
    # 0. Find postings with WFH dictionary terms
    #=====

    # re-generate job posting
    # df["job_id"] = df["seq_id"].str.split("_").str[0]
    # df["job_id"] = df["job_id"].astype(int) 

    # KEY: if the sequences of a job posting are not ordered appropriately we will
    # have to do it at this point (before grouping by job_id)

    # join the text for all job postings
    df_job = df.groupby('job_id')[col_text].apply(join_char.join).reset_index()
    
    # explicitly clean memory
    del df
    gc.collect()

    #=====
    # 1. Find postings with WFH dictionary terms
    #=====

    # lower case all text (we will not apply any more preprocessing)
    df_job["sequence_clean"] = process_map(dm.clean_sequence, 
                                       df_job[col_text],
                                       repeat(punctuation),
                                       max_workers=multiprocessing.cpu_count(), 
                                       chunksize=multiprocessing.cpu_count())

    # drop original text (to save space)
    df_job.drop(columns=[col_text], inplace=True)

    # apply dictionary tagging and save results in dataframe    
    dict_results = process_map(dm.apply_find_regex_matches, 
                               df_job["sequence_clean"],
                               repeat(dict_regex),
                               max_workers=multiprocessing.cpu_count(), 
                               chunksize=multiprocessing.cpu_count())

    dict_matches, dict_positions = list(zip(*dict_results))
    df_job["dict_matches"] = dict_matches
    df_job["dict_positions"] = dict_positions

    # create a column with the number of hits and binary column signaling if there was any match
    df_job["dict_num"] = df_job["dict_matches"].swifter.apply(lambda x: len(x))
    df_job["dict_binary"] = df_job["dict_matches"].swifter.apply(lambda x: False if len(x) == 0 else True)

    # partition the dataset into sequences with and without matches
    df_wfh = df_job[df_job["dict_binary"] == True]
    df_wfh.reset_index(drop=True, inplace=True)

    df_no_wfh = df_job[df_job["dict_binary"] == False]
    df_no_wfh.reset_index(drop=True, inplace=True)
    print("Dictionary tagging applied!")

    #=================
    # 2. Identify negations (only for postings with WFH terms)
    #=================

    # parameters for negation
    neg_terms_before = list(df_neg["term"])
    neg_terms_after = ["no", "not"]
    window_before = 3
    window_after = 2
    search_nt = True
   
    # apply binary negation tagging (WITH POSITIONS OF NEGATION)
    negation_results = process_map(dm.find_negation_with_terms, 
                                     df_wfh.to_dict(orient='records'),
                                     repeat("sequence_clean"),
                                     repeat(neg_terms_before), 
                                     repeat(neg_terms_after),
                                     repeat(window_before), 
                                     repeat(window_after),
                                     repeat(search_nt), 
                                     repeat(tokenization_pattern),
                                     max_workers=multiprocessing.cpu_count(), 
                                     chunksize=multiprocessing.cpu_count())
   
    neg_binary, neg_matches = list(zip(*negation_results))
    df_wfh["negation"] = neg_binary
    df_wfh["neg_matches"] = neg_matches
    print("Negation tagging applied!")

    #=====
    # 3. Find postings with generic terms (only for postings without WFH terms)
    #=====

    # apply regex search
    generic_results = process_map(dm.apply_find_regex_matches,
                                  df_no_wfh["sequence_clean"],
                                  repeat(generic_regex),
                                  max_workers=multiprocessing.cpu_count(), 
                                  chunksize=multiprocessing.cpu_count()
                                  )

    generic_matches, generic_positions = list(zip(*generic_results))
    df_no_wfh["generic_matches"] = generic_matches
    df_no_wfh["generic_positions"] = generic_positions
    df_no_wfh["num_generic"] = df_no_wfh["generic_matches"].swifter.apply(lambda x: len(x))
    df_no_wfh["binary_generic"] = df_no_wfh["generic_matches"].swifter.apply(lambda x: False if len(x) == 0 else True)

    #=====
    # 4. Consolidate dataframes
    #=====

    # for postings with WFH terms we have two categories: non-negated and negated
    df_wfh["non_negated_wfh"] = ~df_wfh["negation"]
    df_wfh.rename(columns={"negation": "negated_wfh"}, inplace=True)
    #df_wfh = df_wfh[["seq_id", "sequence_clean", "non_negated_wfh", "negated_wfh", "dict_matches", "dict_positions"]]
    
    # for postings without WFH terms we have two categories: generic and non-generic
    df_no_wfh["non_generic"] = ~df_no_wfh["binary_generic"]
    df_no_wfh.rename(columns={"binary_generic": "generic_wfh"}, inplace=True)
    #df_no_wfh = df_no_wfh[["seq_id", "sequence_clean", "non_generic", "generic_wfh", "generic_matches", "generic_positions"]]

    # merge dataframes, start by creating missing columns on each dataframe
    df_wfh["generic_wfh"] = False
    df_wfh["num_generic"] = False
    df_wfh["generic_matches"] = np.empty((len(df_wfh), 0)).tolist()
    df_wfh["generic_positions"] = np.empty((len(df_wfh), 0)).tolist()
    df_wfh["non_generic"] = False

    df_no_wfh["non_negated_wfh"] = False
    df_no_wfh["negated_wfh"] = False
    df_no_wfh["neg_matches"] = np.empty((len(df_no_wfh), 0)).tolist()

    df_final = pd.concat([df_wfh, df_no_wfh], axis=0)
    df_final.reset_index(drop=True, inplace=True)
    df_final

    # verify that groups are mutually exclusive
    assert(df_final["non_negated_wfh"].sum() + df_final["negated_wfh"].sum() + df_final["non_generic"].sum() + df_final["generic_wfh"].sum() == len(df_final))

    # explicitly clean memory
    del generic_results
    del generic_matches
    del generic_positions
    del dict_results
    del dict_matches
    del dict_positions
    del negation_results
    del neg_matches
    del neg_binary
    del df_job
    del df_no_wfh
    del df_wfh
    gc.collect()

    # return tagged data
    return df_final

def get_group_name(row):
    if row['negated_wfh'] == 1:
        return 'negated_wfh'
    elif row['non_negated_wfh'] == 1:
        return 'non_negated_wfh'
    elif row['generic_wfh'] == 1:
        return 'generic_wfh'
    elif row['non_generic'] == 1:
        return 'non_generic'
    else:
        return np.nan  # In case none of the columns has a 1
#%%

#=====
# MAIN: execute everything
#=====

def main(country, col_text):

    # define main paths
    sequences_path = config[f"{country}_sequences"]
    data_path = f"{config['bucket_mount_point']}{sequences_path}"

    all_files = os.listdir(data_path)
    all_files = [file for file in all_files if ".parquet" in file]
    print(f"All available files: {len(all_files)}")

    # remove from list the files that have already been processed
    processed_files_local = os.listdir(output_path + f"{country}/")
    processed_files_local = [file for file in processed_files_local]
    processed_files_bucket = os.listdir(config['bucket_mount_point'] + config["output_data"]["sample_selection_tagging"] + f"{country}/")
    processed_files_bucket = [file for file in processed_files_bucket]
    processed_files = processed_files_local + processed_files_bucket
    all_files = [file for file in all_files if file not in processed_files]

    print(f"Total files to process: {len(all_files)}")

    # choose a random file to process
    file = random.choice(all_files)
    print(f"Processing file: {file}")
    start = time.perf_counter()

    # report memory usage at the start
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024 * 1024)} MB")

    # read data from file
    df = pd.read_parquet(data_path + file)

    # generate final data with partition
    df_final = partition_df(df, col_text)

    # make sure all elements of lists are of the same type 
    df_final['neg_matches'] = df_final['neg_matches'].apply(
        lambda lists: [tuple(map(str, tup)) if isinstance(tup, tuple) else str(tup) for tup in lists] if isinstance(lists, list) else lists
    )

    # combine group identifier into a single variable
    df_final['lang_group'] = df_final.apply(get_group_name, axis=1)
    df_final.drop(["sequence_clean", 'negated_wfh', 'non_negated_wfh', 'generic_wfh', 'non_generic'], axis=1, inplace=True)
    # Verify there are no missing values in the new 'group' column
    assert df_final['lang_group'].isnull().sum() == 0, "There are missing values in the 'group' column"

    # save data in parquet format
    file_save_name = file
    df_final.to_parquet(output_path + f"{country}/" + file_save_name, index=False)

    # Explicitly clean up memory
    del df
    del df_final
    gc.collect()

    # report memory usage at the end
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024 * 1024)} MB")
