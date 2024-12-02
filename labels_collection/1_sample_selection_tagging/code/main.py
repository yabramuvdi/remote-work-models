# standard libraries
import os
import yaml
import subprocess
from multiprocessing import Process

# import the code from all the other scripts
import my_config
import sampling_groups

# main paths
input_path = "../input/"
temp_path = "../temp/"
output_path = "../output/"
config_path = "../config/"


if __name__ == '__main__':

    #=====
    # 0. Setup
    #=====

    # load parameters
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

    #=====
    # 1. Loop through all countries to perform tagging
    #=====

    #countries = ["us", "uk", "can", "anz"]
    countries = ["uk"]
    for country in countries:
        
        print(f"Processing files for country: {country.upper()}")
        
        # get the number of files to process for each country
        num_iters = my_config.main(country)
        num_iters = 1

        for _ in range(num_iters):
            # tag postings
            sampling_groups.main(country, col_text="sequence")
            
            # another approach
            #p = Process(target=sampling_groups.main, args=(country, "sequence"))
            #p.start()
            #p.join()
    
    # check if temp directory exists
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)  # create directory if it does not exist

    # create a file indicating that the process is ready
    with open(temp_path + "ready.txt", "w") as f:
        f.write("File created")

    #=====
    # 2. Transfer results to bucket
    #=====
    
    # bash command (example)
    #gsutil cp -n -r /home/yabra/sampling/output/* gs://for_transfer/remote-work-en/tasks/model_creation/1_sample_selection_tagging/output/
    
    #command = f"gsutil cp -n -r {output_path}* gs://{config['bucket_name']}/{config['task_path']}output/" 
    #result = subprocess.run(command, capture_output=True, text=True, shell=True)