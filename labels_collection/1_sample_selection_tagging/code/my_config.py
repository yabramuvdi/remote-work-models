import os
import argparse
import yaml
import subprocess
import argparse

def main(country):

    # main paths
    input_path = "../input/"
    temp_path = "../temp/"
    output_path = "../output/"
    config_path = "../config/"

    # load parameters
    with open(config_path + "config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # define main paths
    sequences_path = config[f"{country}_sequences"]
    data_path = f"{config['bucket_mount_point']}{sequences_path}"
    output_path = f"{output_path}{country.lower()}/"

    # get all available files
    all_files = os.listdir(data_path)
    all_files = [file for file in all_files if ".parquet" in file]
    print(f"All available files: {len(all_files)}")

    # remove from list the files that have already been processed
    processed_files_local = os.listdir(output_path)
    processed_files_local = [file for file in processed_files_local]
    processed_files_bucket = os.listdir(config['bucket_mount_point'] + config["output_data"]["sample_selection_tagging"] + f"{country}/")
    processed_files_bucket = [file for file in processed_files_bucket]
    processed_files = processed_files_local + processed_files_bucket
    all_files = [file for file in all_files if file not in processed_files]
    print(f"Total files to process: {len(all_files)}")

    num_runs = len(all_files)
    print(f"Number of files to process: {num_runs}")

    return num_runs