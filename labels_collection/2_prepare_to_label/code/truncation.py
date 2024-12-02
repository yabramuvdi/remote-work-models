#%%

import pandas as pd
import yaml
import os
import subprocess
import pyreadr
import random
import re
from tqdm import tqdm

#========
# 0. Setup
#========

# main paths
input_path = "../input/"
output_path = "../output/"
config_path = "../config/"

# load config options
with open(config_path + "config.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# load parameters
with open(config_path + "params.yaml", "r") as stream:
    try:
        params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# mount the bucket (bash command)
command = f"gcsfuse --implicit-dirs {config['bucket_name']} {config['bucket_mount_point']}" 
result = subprocess.run(command, capture_output=True, text=True, shell=True)

if result.returncode != 0:
    print(f"Error occurred: {result.stderr}")
else:
    print(f"Output: {result.stdout}")


tagged_path = f"{config['bucket_mount_point']}{config['output_data']['sample_selection_tagging']}"

#%%

#=======
# Functions
#=======

def read_r(file_path):
    result = pyreadr.read_r(file_path)
    df_temp = result[None].copy()
    #return df_temp[["seq_id", "sequence"]]
    return df_temp

def wfh_truncate(row):
        """ Find the first appearance of a specific WFH term. Construct a 512 words window around it.
            If a symmetric window is possible, use it. If not, use an asymmetric window with
            start or end of job posting as a boundary.
        """
        
        word_limit = int(512/words2tokens)

        # get the information of the first match
        match_term = row["dict_matches"][0]
        first_positions = row['dict_positions'][0]
        num_words_match = len(match_term.split())

        # split the posting into the text before and the text after the match
        text_before = row["sequence"][0:first_positions[0]]
        text_after = row["sequence"][first_positions[1]:]

        # count the number of words before and after
        #text_before_words = text_before.split()
        text_before_words = re.findall(r"\S+|\n", text_before)
        num_words_before = len(text_before_words)

        #text_after_words = text_after.split()
        text_after_words = re.findall(r"\S+|\n", text_after)
        num_words_after = len(text_after_words)

        # chek if a symmetric window is possible
        if num_words_before >= (word_limit//2 + num_words_match) and num_words_after >= (word_limit//2 + num_words_match):
            truncated_before = " ".join(text_before_words[-word_limit//2:])
            truncated_after = " ".join(text_after_words[:word_limit//2:])
            return truncated_before + " " + match_term + " " + truncated_after
        # if not, check if we must start from the begining
        elif num_words_before < (word_limit//2 + num_words_match):
            #posting_words = row["sequence"].split()
            posting_words = re.findall(r"\S+|\n", row["sequence"])
            return " ".join(posting_words[0:word_limit])
        else:
            #posting_words = row["sequence"].split()
            posting_words = re.findall(r"\S+|\n", row["sequence"])
            return " ".join(posting_words[-word_limit:])


def negated_wfh_truncate(row):
    """ Find the first appearance of a negated WFH term. Construct a 512 words window around it.
        If a symmetric window is possible, use it. If not, use an asymmetric window with
        start or end of job posting as a boundary.
    """
    
    word_limit = int(512/words2tokens)

    # get the information of the first negation
    negated_term_pos = int(row["neg_matches"][0][1])

    # get the information of the negated match
    match_term = row["dict_matches"][negated_term_pos]
    first_positions = row['dict_positions'][negated_term_pos]
    num_words_match = len(match_term.split())

    # split the posting into the text before and the text after the match
    text_before = row["sequence"][0:first_positions[0]]
    text_after = row["sequence"][first_positions[1]:]

    # count the number of words before and after
    #text_before_words = text_before.split()
    text_before_words = re.findall(r"\S+|\n", text_before)
    num_words_before = len(text_before_words)

    #text_after_words = text_after.split()
    text_after_words = re.findall(r"\S+|\n", text_after)
    num_words_after = len(text_after_words)

    # chek if a symmetric window is possible
    if num_words_before >= (word_limit//2 + num_words_match) and num_words_after >= (word_limit//2 + num_words_match):
        truncated_before = " ".join(text_before_words[-word_limit//2:])
        truncated_after = " ".join(text_after_words[:word_limit//2:])
        return truncated_before + " " + match_term + " " + truncated_after
    # if not, check if we must start from the begining
    elif num_words_before < (word_limit//2 + num_words_match):
        #posting_words = row["sequence"].split()
        posting_words = re.findall(r"\S+|\n", row["sequence"])
        return " ".join(posting_words[0:word_limit])
    else:
        #posting_words = row["sequence"].split()
        posting_words = re.findall(r"\S+|\n", row["sequence"])
        return " ".join(posting_words[-word_limit:])


def generic_truncate(row):
    """ Find the first appearance of a generic term. Construct a 512 words window around it.
        If a symmetric window is possible, use it. If not, use an asymmetric window with
        start or end of job posting as a boundary.
    """
    
    word_limit = int(512/words2tokens)

    # get the information of the first match
    match_term = row["generic_matches"][0]
    first_positions = row['generic_positions'][0]
    num_words_match = len(match_term.split())

    # split the posting into the text before and the text after the match
    text_before = row["sequence"][0:first_positions[0]]
    text_after = row["sequence"][first_positions[1]:]

    # count the number of words before and after
    #text_before_words = text_before.split()
    text_before_words = re.findall(r"\S+|\n", text_before)
    num_words_before = len(text_before_words)

    #text_after_words = text_after.split()
    text_after_words = re.findall(r"\S+|\n", text_after)
    num_words_after = len(text_after_words)

    # chek if a symmetric window is possible
    if num_words_before >= (word_limit//2 + num_words_match) and num_words_after >= (word_limit//2 + num_words_match):
        truncated_before = " ".join(text_before_words[-word_limit//2:])
        truncated_after = " ".join(text_after_words[:word_limit//2:])
        return truncated_before + " " + match_term + " " + truncated_after
    # if not, check if we must start from the begining
    elif num_words_before < (word_limit//2 + num_words_match):
        #posting_words = row["sequence"].split()
        posting_words = re.findall(r"\S+|\n", row["sequence"])
        return " ".join(posting_words[0:word_limit])
    else:
        #posting_words = row["sequence"].split()
        posting_words = re.findall(r"\S+|\n", row["sequence"])
        return " ".join(posting_words[-word_limit:])

def non_generic_truncate(row):
    """ Get the first 512 tokens
    """

    word_limit = int(512/words2tokens)
    posting_words = re.findall(r"\S+|\n", row["sequence"])
    return " ".join(posting_words[:word_limit])


#%%

#==============
# Truncate long postings and save
#==============

# read the information from postings sampled
df_sample = pd.read_csv(output_path + "sampled_postings.csv")

#%%

#### iterate over each country/file to read the raw sequences
df_sample_complete = pd.DataFrame()
for file, country in tqdm(df_sample.groupby(["file_name", "country"]).size().index):

    print(f"Processing file {file} for {country.upper()}")
    df_file = df_sample.loc[(df_sample["file_name"] == file) & (df_sample["country"] == country)]

    # define paths to raw sequences
    sequences_path = config["global_data"][f"{country}_sequences"]
    country_data_path = f"{config['bucket_mount_point']}{sequences_path}"
    file_raw = file + ".rds"

    # load all data files
    df_raw = read_r(country_data_path + file_raw)
    df_raw["job_id"] = df_raw["job_id"].astype(int) 

    # select only from the raw data the postings we need
    df_raw = df_raw.loc[df_raw["job_id"].isin(df_file["job_id"])]
    # join the text for all job postings
    df_job = df_raw.groupby('job_id')["sequence"].apply(" ".join).reset_index()
    
    # merge both dataset
    df_job_info = pd.merge(df_file, df_job, on="job_id", how="inner")

    # read tagged file
    file_tagged = file + ".parquet"
    df_tagged = pd.read_parquet(tagged_path + f"/{country}/{file_tagged}")
    df_tagged = df_tagged.loc[df_tagged["job_id"].isin(df_file["job_id"])]
    
    df_job_info = pd.merge(df_job_info, df_tagged, on="job_id", how="inner")

    # convert words to BERT tokens
    words2tokens = 1.46
    token_limit = 512

    # generate word count and token count
    df_job_info['word_count'] = df_job_info['sequence'].apply(lambda x: len(str(x).split()))
    df_job_info["token_count"] = df_job_info["word_count"]*words2tokens + 1
    df_job_info["token_count"] = df_job_info["token_count"].astype(int)

    # split data into those postings that need to be truncated and those that don't
    df_short = df_job_info.loc[df_job_info["token_count"] <= token_limit]
    df_long = df_job_info.loc[df_job_info["token_count"] > token_limit]
    assert(len(df_job_info) == (len(df_short) + len(df_long)))

    # split the dataframe into groups
    df_generic = df_long.loc[df_long["lang_partition"] == "generic_wfh"]
    df_wfh = df_long.loc[df_long["lang_partition"] == "non_negated_wfh"]
    df_negated_wfh = df_long.loc[df_long["lang_partition"] == "negated_wfh"]
    df_non_generic = df_long.loc[df_long["lang_partition"] == "non_generic"]

    # apply the correct function for truncation for each one
    if len(df_long) > 0:
        if len(df_generic) > 0:
            df_generic["truncated_text"] = df_generic.apply(generic_truncate, axis=1)

        if len(df_wfh) > 0:
            df_wfh["truncated_text"] = df_wfh.apply(wfh_truncate, axis=1)

        if len(df_negated_wfh) > 0:
            df_negated_wfh["truncated_text"] = df_negated_wfh.apply(negated_wfh_truncate, axis=1)
        
        if len(df_non_generic) > 0:
            df_non_generic["truncated_text"] = df_non_generic.apply(non_generic_truncate, axis=1)

        # save a consolidated dataframe with the truncated examples
        df_truncated = pd.concat([df_negated_wfh, df_wfh, df_generic, df_non_generic])
        assert(len(df_truncated) == len(df_long))
        #df_truncated.rename(columns={"sequence": "posting"}, inplace=True)
        df_truncated.reset_index(drop=True, inplace=True)

        # unify data types to save as parquet
        df_truncated['neg_matches'] = df_truncated['neg_matches'].apply(lambda d: d if isinstance(d, list) else [])
        df_truncated['dict_matches'] = df_truncated['dict_matches'].apply(lambda d: d if isinstance(d, list) else [])
        df_truncated['dict_positions'] = df_truncated['dict_positions'].apply(lambda d: d if isinstance(d, list) else [])
        df_truncated['generic_matches'] = df_truncated['generic_matches'].apply(lambda d: d if isinstance(d, list) else [])
        df_truncated['generic_positions'] = df_truncated['generic_positions'].apply(lambda d: d if isinstance(d, list) else [])

        # convert all list columns just to strings
        df_truncated['neg_matches'] = df_truncated['neg_matches'].apply(str)
        df_truncated['dict_matches'] = df_truncated['dict_matches'].apply(str)
        df_truncated['dict_positions'] = df_truncated['dict_positions'].apply(str)
        df_truncated['generic_matches'] = df_truncated['generic_matches'].apply(str)
        df_truncated['generic_positions'] = df_truncated['generic_positions'].apply(str)

    # join truncated and not truncated postings
    df_short["truncated_text"] = ""

    # save to global dataframe the relevant data
    for d in [df_truncated, df_short]:
        if len(d) > 0:
            df_sample_complete = pd.concat([df_sample_complete,
                                            d[["job_id", "sequence", "truncated_text"]]
                                            ])
#%%

# save final data
df_sample_complete.rename(columns={"sequence": "original_posting"}, inplace=True)
df_sample_complete[["job_id", "original_posting", "truncated_text"]].to_parquet(output_path + "sampled_postings_complete.parquet", index=False)
# %%
