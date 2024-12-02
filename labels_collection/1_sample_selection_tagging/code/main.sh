#!/bin/bash

while [ ! -f ../temp/ready.txt ]; do
    python3 main.py
    # optional: sleep to prevent too much CPU usage
    sleep 5
done

#==================
# OLD APPROACH
#==================

# # No need to mount the bucket (the python script does it)
# #gcsfuse --implicit-dirs for_transfer /home/yabra/data

# # run script to setup directories
# #python3 ../

# # define a function to stop all the code when hitting ctrl-c
# trap ctrl_c INT

# function ctrl_c() {
#         echo "** Trapped CTRL-C"
#         exit
# }

# # define the countries
# countries=("us" "uk" "can" "anz")

# for country in "${countries[@]}"; do
    
#     # Get the number of times to run the Python script
#     python3 my_config.py $country
#     # Read the number of loops from the file
#     num_runs=$(cat ../temp/num_runs.txt)
#     echo "Starting process for $country for $num_runs iterations"

#     # Run the Python script the specified number of times
#     for ((i = 1; i <= num_runs; i++)); do
#       echo "Running script for $country, iteration: $i"
#       python3 sampling_groups.py $country
#     done

#     # Transfer results to bucket
#     #gsutil cp -n -r /home/yabra/sampling/output/test/* gs://for_transfer/remote-work-en/tasks/model_creation/1_sample_selection_tagging/output/

# done
