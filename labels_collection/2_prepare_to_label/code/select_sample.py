#%%

import pandas as pd
import yaml
import os
import subprocess
import pyreadr
import random

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


# %%

#========
# Sample selection
#========

# create a dataframe to store sampled postings
df_sample = pd.DataFrame()
sample_size = params["total_postings"]
relevant_countries = {key:value for key,value in params["country_shares"].items() if value > 0}

#### country level for loop
for country, country_share in relevant_countries.items():

    df_sample_country = pd.DataFrame()

    num_country = round(sample_size*country_share)
    print("-"*1 + f" Draw {num_country} postings from {country}")

    sequences_path = config["global_data"][f"{country}_sequences"]
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
    for time_partition, time_share in params["time_shares"].items():

        if time_share > 0: 
            num_partition = round(num_country*time_share)
            print("-"*4 + f" Draw {num_partition} postings from {time_partition}")
            df_partition = df_files.loc[df_files["date_category"] == time_partition].copy()
            df_partition['quarter'] = df_partition['date'].dt.quarter

            # sample uniformly from each quarter
            for quarter in df_partition["quarter"].unique():
                
                num_quarter = round(num_partition/len(df_partition["quarter"].unique()))
                print("-"*8 + f" Draw {num_quarter} postings from Q{quarter}")

                # read all tagged files from the quarter
                df_quarter = df_partition.loc[df_partition["quarter"] == quarter]
                print("-"*8 + f" Loading all available files: {len(df_quarter)}")
                df_tagged_quarter = pd.DataFrame()
                
                for file in df_quarter["file_name"].values:
                    file_tagged = file + ".parquet"
                    df_tagged = pd.read_parquet(tagged_path + f"/{country}/{file_tagged}")
                    df_tagged["file_name"] = file 
                    df_tagged_quarter = pd.concat([df_tagged_quarter, df_tagged])
                
                print("-"*8 + f" All files of quarter are loaded")

                # read original file
                #file_raw = file + ".rds"
                #df_raw = read_r(country_data_path + file_raw)

                #### language partition loop
                for lang_partition, lang_share in params["language_shares"].items():
                    num_lang = round(num_quarter*lang_share)
                    print("-"*12 + f" Draw {num_lang} postings for language partition {lang_partition}")
                    
                    df_lang = df_tagged_quarter.loc[df_tagged_quarter[lang_partition] == True]
                    
                    # sample for the given language partition and simplify data to append to global df
                    if num_lang <= len(df_lang):
                        df_lang_sample = df_lang.sample(num_lang)
                    else:
                        df_lang_sample = df_lang.sample(frac=1.0)

                    df_lang_sample = df_lang_sample[["job_id", "file_name"]]
                    df_lang_sample["lang_partition"] = lang_partition
                    df_lang_sample["time_partition"] = time_partition
                    df_lang_sample["country"] = country

                    # update the global data
                    df_sample_country = pd.concat([df_sample_country, df_lang_sample])
                    
    # Evaluate the current state of the country sample
    if len(df_sample_country) < num_country:

        # randomly draw a file and a language partition to achieve the final number of postings
        print("-"*4 + f" Draw {num_country - len(df_sample_country)} additional postings at random for {country}")
        while len(df_sample_country) < num_country:
            
            # sample file and partition
            file = df_files[["file_name"]].sample(1)["file_name"].values[0]
            file_tagged = file + ".parquet"
            lang_partition = random.sample(list(params["language_shares"].keys()), 1)[0]
            
            # read tagged file
            df_tagged = pd.read_parquet(tagged_path + f"/{country}/{file_tagged}")
            df_lang = df_tagged.loc[df_tagged[lang_partition] == True]
            df_lang_sample = df_lang.sample(1)
            df_lang_sample = df_lang_sample[["job_id"]]
            df_lang_sample["lang_partition"] = lang_partition
            df_lang_sample["file_name"] = file
            df_lang_sample["time_partition"] = df_files.loc[df_files["file_name"] == file, "date_category"].values[0]
            df_lang_sample["country"] = country
            
            # update the global data and the counter
            df_sample_country = pd.concat([df_sample_country, df_lang_sample])
    
    elif len(df_sample_country) > num_country:
        # sample randomly from the dataframe
        print("-"*4 + f" Remove {len(df_sample_country) - num_country} postings at random for {country}")
        df_sample_country = df_sample_country.sample(num_country)

    # Save contry data
    df_sample = pd.concat([df_sample, df_sample_country])
    del df_sample_country
    print("-"*5 + f" {country.upper()} ready! " + "-"*5)


# save final sample
df_sample.to_csv(output_path + "sampled_postings.csv", index=False)

# %%
