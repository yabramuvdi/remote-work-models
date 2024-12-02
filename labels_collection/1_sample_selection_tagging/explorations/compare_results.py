#%%

import pandas as pd
import yaml
import os
import subprocess
import pyreadr
import random
import numpy as np

#========
# 0. Setup
#========

def read_r(file_path):
    result = pyreadr.read_r(file_path)
    df_temp = result[None].copy()
    #return df_temp[["seq_id", "sequence"]]
    return df_temp

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

# mount the bucket (bash command)
command = f"gcsfuse --implicit-dirs {config['bucket_name']} {config['bucket_mount_point']}" 
result = subprocess.run(command, capture_output=True, text=True, shell=True)

if result.returncode != 0:
    print(f"Error occurred: {result.stderr}")
else:
    print(f"Output: {result.stdout}")


# %%

#=======
# Select relevant files
#=======

# select country
country = "us"

# read all tagged files from the previous task
tagged_path = f"{config['bucket_mount_point']}{config['output_data']['sample_selection_tagging']}"
tagged_files = os.listdir(tagged_path + f"/{country}/")
tagged_files = [file.replace(".parquet", "") for file in tagged_files]

# categorize all files according to their time period
df_files = pd.DataFrame({"file_name": tagged_files})
df_files["date"] = df_files["file_name"].apply(lambda x: x.split("_")[1])
df_files["date"] = pd.to_datetime(df_files["date"], format='%Y%m%d')

def categorize_date(date):
    if date.year < 2018:
        return 'pre_2018'
    elif date.year == 2018:
        return '2018'
    elif date.year == 2019 or (date.year == 2020 and date.month < 3):
        return '2019_to_feb_2020'
    elif date.year == 2020 and date.month >= 3:
        return 'march_to_dec_2020'
    elif date.year == 2021:
        return '2021'
    elif date.year == 2022:
        return '2022'
    elif date.year == 2023:
        return '2023'

df_files['date_category'] = df_files['date'].apply(categorize_date)
df_files

# %%

files_2020 = df_files.loc[df_files["date_category"] == "march_to_dec_2020"]
files_2023 = df_files.loc[df_files["date_category"] == "2023"]

# %%

#======
# By year statistics
#======

#### 2023
year_terms = pd.DataFrame()
for file in files_2023["file_name"].values:
    
    # read tagged file
    file_tagged = file + ".parquet"
    df_file = pd.read_parquet(tagged_path + f"/{country}/{file_tagged}")
    
    # explode the list into separate rows
    exploded_df = df_file[["dict_matches"]].explode('dict_matches')
    exploded_df.dropna(inplace=True)
    year_terms = pd.concat([year_terms, exploded_df])

word_counts = year_terms['dict_matches'].value_counts()
word_counts_dict = word_counts.to_dict()

total_terms_2023 = word_counts.sum()
hybrid_terms_2023 = 0
for term, count in word_counts_dict.items():
    if "hybrid" in term:
        print(term)
        hybrid_terms_2023 += count

print(f"{np.round((hybrid_terms_2023/total_terms_2023)*100, 3)}% of term mention hybrid ({hybrid_terms_2023})")
# %%

### 2020
year_terms = pd.DataFrame()
for file in files_2020["file_name"].values:
    
    # read tagged file
    file_tagged = file + ".parquet"
    df_file = pd.read_parquet(tagged_path + f"/{country}/{file_tagged}")
    
    # explode the list into separate rows
    exploded_df = df_file[["dict_matches"]].explode('dict_matches')
    exploded_df.dropna(inplace=True)
    year_terms = pd.concat([year_terms, exploded_df])

word_counts = year_terms['dict_matches'].value_counts()
word_counts_dict = word_counts.to_dict()

total_terms_2020 = word_counts.sum()
hybrid_terms_2020 = 0
for term, count in word_counts_dict.items():
    if "hybrid" in term:
        print(term)
        hybrid_terms_2020 += count

print(f"{np.round((hybrid_terms_2020/total_terms_2020)*100, 3)}% of term mention hybrid ({hybrid_terms_2020})")

# %%

diff = (hybrid_terms_2023/total_terms_2023)/(hybrid_terms_2020/total_terms_2020)
diff
# %%
