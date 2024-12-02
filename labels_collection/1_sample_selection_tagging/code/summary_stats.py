# Generate some basic summary stats from the assignment process including the group weights

#%%

import pandas as pd
import numpy as np
import yaml
import os
import pyreadr
import subprocess
from tqdm import tqdm

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

#### functions
def read_r(file_path):
    result = pyreadr.read_r(file_path)
    df_temp = result[None].copy()
    #return df_temp[["seq_id", "sequence"]]
    return df_temp

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

#%%
    
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

# # load parameters
# with open(config_path + "params.yaml", "r") as stream:
#     try:
#         params = yaml.safe_load(stream)
#     except yaml.YAMLError as exc:
#         print(exc)

# mount the bucket (bash command)
command = f"gcsfuse --implicit-dirs {config['bucket_name']} {config['bucket_mount_point']}" 
result = subprocess.run(command, capture_output=True, text=True, shell=True)

if result.returncode != 0:
    print(f"Error occurred: {result.stderr}")
else:
    print(f"Output: {result.stdout}")

#%%
    
# create a dataframe to store summary stats
df_global = pd.DataFrame()
relevant_countries = ["us"]
relevant_periods = ["pre_2018", "2018", "2019_to_feb_2020", 
                    "march_to_dec_2020", "2021", "2022", "2023"]

#### country level for loop
for country in relevant_countries:

    print("-"*1 + f" Draw postings from {country}")

    sequences_path = config[f"{country}_sequences"]
    country_data_path = f"{config['bucket_mount_point']}{sequences_path}"

    # read all tagged files from the previous task
    tagged_path = f"{config['bucket_mount_point']}{config['output_data']['sample_selection_tagging']}"
    tagged_files = os.listdir(tagged_path + f"/{country}/")
    tagged_files = [file.replace(".parquet", "") for file in tagged_files]

    # categorize all files according to their time period
    df_files = pd.DataFrame({"file_name": tagged_files})
    df_files["date"] = df_files["file_name"].apply(lambda x: x.split("_")[1])
    df_files["date"] = df_files.apply(lambda x: x["date"] if x["date"] != "" else x["file_name"].split("_")[-1], axis=1)
    df_files["date"] = pd.to_datetime(df_files["date"], format='%Y%m%d')

    # categorize the date
    df_files['date_category'] = df_files['date'].apply(categorize_date)

    #### time period loop
    for time_partition in relevant_periods:

        print("-"*4 + f" Draw postings from {time_partition}")
        df_partition = df_files.loc[df_files["date_category"] == time_partition].copy()

        #### file loop
        for file in tqdm(df_partition["file_name"].values):
            # load the relevant files
            file_tagged = file + ".parquet"
            df_tagged = pd.read_parquet(tagged_path + f"/{country}/{file_tagged}")
            
            # check for the pressence of hybrid
            df_tagged["generic_hybrid"] = df_tagged["generic_matches"].apply(lambda x: "hybrid" in x)

            # calculate the number of postings per language category
            num_negated_wfh = df_tagged["negated_wfh"].sum()
            num_non_neg_wfh = df_tagged["non_negated_wfh"].sum()
            num_generic_wfh = df_tagged["generic_wfh"].sum()
            num_non_generic_wfh = df_tagged["non_generic"].sum()
            assert((num_negated_wfh + num_non_neg_wfh + num_generic_wfh + num_non_generic_wfh) == len(df_tagged))

            # results = {"file": file,
            #            "date": time_partition,
            #            "num_postings": len(df_tagged),
            #            "negated_wfh": num_negated_wfh,
            #            "non_neg_wfh": num_non_neg_wfh,
            #            "generic_wfh": num_generic_wfh,
            #            "non_generic_wfh": num_non_generic_wfh}
            
            num_generic_no_hybrid = len(df_tagged.loc[(df_tagged["generic_wfh"]) & ~(df_tagged["generic_hybrid"])])
            num_generic_hybrid = len(df_tagged.loc[(df_tagged["generic_wfh"]) & (df_tagged["generic_hybrid"])])
            
            assert((num_negated_wfh + num_non_neg_wfh + num_generic_hybrid + num_generic_no_hybrid + num_non_generic_wfh) == len(df_tagged))

            results = {"file": file,
                       "date": time_partition,
                       "num_postings": len(df_tagged),
                       "negated_wfh": num_negated_wfh,
                       "non_neg_wfh": num_non_neg_wfh,
                       "non_generic_wfh": num_non_generic_wfh,
                       "generic_hybrid": num_generic_hybrid,
                       "generic_non_hybrid": num_generic_no_hybrid}
            

            df_results = pd.DataFrame([results])
            df_global = pd.concat([df_global, df_results])

    #             # read original file
    #             #file_raw = file + ".rds"
    #             #df_raw = read_r(country_data_path + file_raw)


#%%

# generate final summary statistics by year
df_dates = df_global.groupby("date").sum()           

# Normalize the columns by 'num_postings'
for column in df_dates.columns:
    if column != 'num_postings':  # Skip the 'num_postings' column
        df_dates[column] = df_dates[column] / df_dates['num_postings']

df_dates.to_csv(output_path + "sampling_groups_weights_hybrid.csv", index=True)

#%%







#%%
        

# #==============
# # Calculate statistics
# #==============

# #### number of hits

# # save distribution of WFH hits given that there is a WFH hit
# dist_wfh_file = df_final.loc[df_final["non_negated_wfh"]][["dict_num"]]

# # save distribution of generic hits given that there is a generic hit
# dist_generic_file = df_final.loc[df_final["generic_wfh"]][["num_generic"]]

# #### distance between hits

# def hit_wfh_distance(row):
#     if row["dict_matches"]:
#         first_hit = row["dict_positions"][0]
#         last_hit = row["dict_positions"][-1]
#         char_dist = last_hit[1] - first_hit[0]
#         return char_dist
#     else:
#         return 0

# def hit_generic_distance(row):
#     if row["generic_matches"]:
#         first_hit = row["generic_positions"][0]
#         last_hit = row["generic_positions"][-1]
#         char_dist = last_hit[1] - first_hit[0]
#         return char_dist
#     else:
#         return 0
    
# # apply functions
# df_final["distance_wfh"] = df_final.apply(lambda x: hit_wfh_distance(x), axis=1)
# df_final["distance_generic"] = df_final.apply(lambda x: hit_generic_distance(x), axis=1)

# # save distribution of WFH hits given that there is a WFH hit
# dist_wfh_file["distance_wfh"] = df_final.loc[df_final["non_negated_wfh"], "distance_wfh"]
# dist_wfh = pd.read_csv(output_path + "summary_stats/wfh_distribution.csv")
# dist_wfh = pd.concat([dist_wfh, dist_wfh_file])
# dist_wfh.to_csv(output_path + "summary_stats/wfh_distribution.csv", index=False)

# # save distribution of generic hits given that there is a generic hit
# dist_generic_file["distance_generic"] = df_final.loc[df_final["generic_wfh"], "distance_generic"]
# dist_generic = pd.read_csv(output_path + "summary_stats/generic_distribution.csv")
# dist_generic = pd.concat([dist_generic, dist_generic_file])
# dist_generic.to_csv(output_path + "summary_stats/generic_distribution.csv", index=False)

# %%
