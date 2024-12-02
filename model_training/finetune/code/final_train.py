"""
Fine-tuning a language model (we use DistilBERT) with the optimal set of parameters
using all the available labeled data (train + test). The fine-tuned model will be stored
in: ./models/fine-tune/final/
"""

#%%

import pandas as pd
import numpy as np
import yaml
from transformers import DistilBertTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from transformers import TrainingArguments
from transformers import Trainer
import torch
import sys
import os

# load utils functions
sys.path.insert(0, "../utils/")
import nlp_utils as nlp_utils

print(f"GPU: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

data_path = "../data/"
hand_path = "./"

pretrain_path = "distilbert-base-uncased"           # if we want to use a standard model
#pretrain_path = "../models/pre-train/final/"         # if we want to use the pretrained model

output_path = "../models/fine-tune/"
checkpoint_path = "../models/fine-tune/checkpoints/"
final_model_path = "../models/fine-tune/final/"

# check if the output directories exist and, if not, create them
for dir_path in [output_path, checkpoint_path, final_model_path]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory {dir_path} created successfully")
    else:
        print(f"Directory {dir_path} already exists")

# open YAML file with parameters for data consolidation
stream = open(hand_path + "cv_distilbert_params.yaml", 'r')
params = yaml.load(stream, Loader=yaml.Loader)

# %%

#=============================
# Data and tokenization
#=============================

# # [OLD:_ Amazon Reviews] load all data
# df = pd.read_csv(data_path + "Reviews.csv")
# df = df.sample(100)

# # make target variable binary and integer
# # KEY: the name "label" is important. If we use a different name
# # we need to tell the model the name of our target variable
# df["label"] = df["Score"].apply(lambda x: 1 if x > 3 else 0)
# df

#%%

# load all training data from AMT
df_core_train = pd.read_csv(data_path + "train_core_amt.csv")
df_core_train = df_core_train[["sequence_id", "sequence", "label"]]

# load supplementary training data
df_sup_train = pd.read_csv(data_path + "train_supplementary.csv")

# join both datasets and make sure label is an integer
df_train = pd.concat([df_core_train, df_sup_train])
df_train["label"] = df_train["label"].astype(int)

# load test data
df_test = pd.read_csv(data_path + "test_core_amt.csv")
df_test = df_test[["sequence_id", "sequence", "label"]]
df_test["label"] = df_test["label"].astype(int)

# join all datasets together and shuffle the examples
df = pd.concat([df_train, df_test])
df = df.sample(frac=1.0)

# sample for testing
df = df.sample(100)

#%%

# load tokenizer from pre-trained model
tokenizer = DistilBertTokenizer.from_pretrained(pretrain_path, padding=True)

# transform into Dataset class
dataset = Dataset.from_pandas(df)

# tokenize text from both datasets
max_sent_size = params["max_sent_size"] 
tokenized_data = dataset.map(nlp_utils.tokenize_function, 
                             fn_kwargs={"tokenizer": tokenizer, "col_text": "sequence", "max_sent_size": max_sent_size}, 
                             batched=True)

#%%

#=============================
# Optimal parameters
#=============================

# get the parameters with the best average F1 score across the KFolds
df_results = pd.read_csv(output_path + "cv_results.csv")
df_params = df_results.groupby(["epochs", 
                                "learning_rate", 
                                "batch_size", 
                                "max_seq_len"]).agg({"f1": np.mean,
                                                     "accuracy": np.mean})

metric = "f1"
best_model = df_params.loc[df_params[metric] == df_params[metric].max()]

# get the parameters of the model
model_params = {"batch_size": best_model.index.get_level_values("batch_size")[0], 
                "learning_rate": best_model.index.get_level_values("learning_rate")[0],
                "epochs": best_model.index.get_level_values("epochs")[0]}

print(f"Training model configuration: {model_params}")

#%%

#=============================
# Training
#=============================

# load the model
num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(pretrain_path, num_labels=num_labels)

# define all training parameters
training_args = TrainingArguments(checkpoint_path,
                                learning_rate=float(model_params["learning_rate"]),
                                num_train_epochs=int(model_params["epochs"]),
                                per_device_train_batch_size=int(model_params["batch_size"]),
                                evaluation_strategy="no",
                                save_strategy="no"
                                )

# by default the Trainer will use CrossEntropy loss from (torch.nn)
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=nlp_utils.compute_metrics,
    train_dataset=tokenized_data,
    tokenizer=tokenizer
)

# train model
trainer.train()

# save final version of the model
trainer.save_model(final_model_path)

# %%
