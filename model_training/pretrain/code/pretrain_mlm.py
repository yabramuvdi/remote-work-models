"""

Perform additional pre-training on a language model (we will use DistilBERT) through 
the Masked Language Modeling (MLM) task. The pre-trained model will be stored
in: ./models/pre-train/final/

Some additional resources:
- https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling
- https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py
- https://www.thepythoncode.com/article/pretraining-bert-huggingface-transformers-in-python

"""

#%%

import sys
import os
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
import yaml
from datasets import Dataset
import torch
import time

# load utils functions
sys.path.insert(0, "../utils/")
import nlp_utils as nlp_utils


print(f"GPU: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

data_path = "../data/"
hand_path = "./"

output_path = "../models/pre-train/"
checkpoint_path = "../models/pre-train/checkpoints/"
final_model_path = "../models/pre-train/final/"

# check if the output directories exist and, if not, create them
for dir_path in [output_path, checkpoint_path, final_model_path]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory {dir_path} created successfully")
    else:
        print(f"Directory {dir_path} already exists")

# open YAML file with parameters for data consolidation
stream = open(hand_path + "distilbert_params.yaml", 'r')
params = yaml.load(stream, Loader=yaml.Loader)

# %%

#=============================
# 0. Consolidate data
#=============================

# unpack params from the yaml file
test_frac = params["test_frac"]
seed = params["seed"]

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
df = df[["sequence_id", "sequence"]]
df = df.sample(frac=1.0)

# transform into Dataset class
raw_datasets = Dataset.from_pandas(df)

# split into train/test
raw_datasets = raw_datasets.train_test_split(test_size=test_frac, seed=seed)

# clean memory
del df
print(f"Sequences for training: {len(raw_datasets['train'])}")
print(f"Sequences for testing: {len(raw_datasets['test'])}")

#%%

#=============================
# 1. Load model and tokenizer
#=============================

# tokenize text
model_name = params["model_name"]
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForMaskedLM.from_pretrained(model_name)

# if we want to get more details about the model
# print(model.config)  
#%%

#=============================
# 2. Tokenization
#=============================

print(f"Initial vocab size: {len(tokenizer)}")
padding = True
max_sent_size = params["max_sent_size"] 

# add new tokens to the tokenizer
tokenizer.add_tokens("\n")
print(f"Final vocab size: {len(tokenizer)}")
# add random word embeddings to new tokens in the model 
model.resize_token_embeddings(len(tokenizer)) 

# verify that new line gets a token
print(tokenizer("\n \n"))

# apply tokenization to the complete dataset
start_token = time.perf_counter()
tokenized_datasets = raw_datasets.map(
    nlp_utils.tokenize_function,
    fn_kwargs={"tokenizer": tokenizer, 
               "col_text": "sequence",
               "max_sent_size": max_sent_size},
    batched=True,
    #num_proc=32, # num vCPUs
    #remove_columns=["frag"],
    #desc="Running tokenizer on dataset line_by_line",
)

del raw_datasets
duration_token = (time.perf_counter() - start_token)/60
print(f"Tokenization ready in: {duration_token} minutes")
#%%

#=============================
# 3. Prepare Masking
#=============================

# initialize collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm_probability= params["mlm_probability"], # probability of replacing a token by [MASK]
    pad_to_multiple_of=8 if params["pad_to_multiple_of_8"] else None,
)

# data will only get "collated" as the model trains
print(data_collator)
#%%

#=============================
# 4. Training
#=============================

# consolidate training arguments
learning_rate = float(params["learning_rate"])
batch_size = int(params["batch_size"])

training_args = TrainingArguments(checkpoint_path,
                                learning_rate=learning_rate,
                                per_device_train_batch_size=batch_size,
                                per_device_eval_batch_size=batch_size ,
                                max_steps=params["steps"],
                                warmup_ratio=params["warmup_ratio"],
                                evaluation_strategy="steps",
                                eval_steps= params["eval_steps"],
                                save_strategy="no",
                                #save_steps=25000,
                                #save_total_limit=1,
                                logging_dir=checkpoint_path,
                                logging_strategy="steps",
                                logging_steps=params["logging_steps"]
                                )

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,    # use if we want to save the tokenizer as well
    data_collator=data_collator,
)

#%%

# TRAIN!
start_train = time.perf_counter()
checkpoint = None

train_result = trainer.train(resume_from_checkpoint=checkpoint)
train_metrics = train_result.metrics

duration_train = (time.perf_counter() - start_train)/60
print(f"Training finished in: {duration_train} minutes")

# save final model
trainer.save_model(final_model_path)
# %%
