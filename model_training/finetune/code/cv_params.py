"""
Find the optimal parameters for fine-tuning a language model (we use DistilBERT) by 
performing k-fold cross-validation on the training data. Generate predictions on the
test data by using the parameters with the highest average F1 score across data splits.
"""

#%%

import pandas as pd
import numpy as np
import time
import yaml
from sklearn.model_selection import ParameterGrid, train_test_split, KFold
from transformers import DistilBertTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import sys
import os
from datasets import Dataset

import torch

# load utils functions
sys.path.insert(0, "../utils/")
import nlp_utils as nlp_utils

print(f"GPU: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

data_path = "../data/"
hand_path = "./"

#pretrain_path = "distilbert-base-uncased"              # if we want to use a standard model
pretrain_path = "../models/pre-train/final/"            # if we want to use the pretrained model

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

#%%

#=============================
# 0. Consolidate data
#=============================

# load all training data from AMT
df_core_train = pd.read_csv(data_path + "train_core_amt.csv")
df_core_train = df_core_train[["sequence_id", "sequence", "label"]]

# load supplementary training data
df_sup_train = pd.read_csv(data_path + "train_supplementary.csv")

# join both datasets and make sure label is an integer
df_train = pd.concat([df_core_train, df_sup_train])
df_train["label"] = df_train["label"].astype(int)
df_train

# sample for quick development
#df_train = df_train.sample(500)
#df_train.reset_index(inplace=True, drop=True)
#%%

# load test data
df_test = pd.read_csv(data_path + "test_core_amt.csv")
df_test = df_test[["sequence_id", "sequence", "label"]]
df_test["label"] = df_test["label"].astype(int)

# sample for quick development
#df_test = df_test.sample(10)
#df_test.reset_index(inplace=True, drop=True)

#%%

print(f"Train sequences: {len(df_train)} \nTest sequences: {len(df_test)}")

#%%

#=============================
# 1. Prepare K-Fold splits
#=============================

# shuffle examples
df_train = df_train.sample(frac=1.0, random_state=params["seed"])

# generate a KFold object
kf = KFold(n_splits=params["n_splits"], 
           shuffle=True, 
           random_state=params["seed"])

# mini-test: get the data for each split
for split_idx, (train_index, val_index) in enumerate(kf.split(df_train)):
    print(f"Split {split_idx}"  ,"TRAIN:", len(train_index), "VALIDATION:", len(val_index))
    X_train, X_val = df_train.iloc[train_index], df_train.iloc[val_index]

#%%

#=============================
# 2. Tokenizer
#=============================

# load tokenizer from pre-trained model
max_sent_size = params["max_sent_size"] 
tokenizer = DistilBertTokenizer.from_pretrained(pretrain_path, padding=True)

#%%

#=============================
# 3. Cross-validation
#=============================

# store global results
df_results = pd.DataFrame()

# iterate over the train/validation splits
for split_idx, (train_index, eval_index) in enumerate(kf.split(df_train)):

    start = time.perf_counter()
    print(f"KFold {split_idx}")

    # subset the data
    X_train, X_eval = df_train.iloc[train_index], df_train.iloc[eval_index]

    # transform into Dataset class
    dataset_train = Dataset.from_pandas(X_train)
    dataset_eval = Dataset.from_pandas(X_eval)

    # tokenize text from both datasets
    tokenized_train = dataset_train.map(nlp_utils.tokenize_function, 
                                        fn_kwargs={"tokenizer": tokenizer, "col_text": "sequence", "max_sent_size": max_sent_size}, 
                                        batched=True)

    tokenized_eval = dataset_eval.map(nlp_utils.tokenize_function, 
                                      fn_kwargs={"tokenizer": tokenizer, "col_text": "sequence", "max_sent_size": max_sent_size}, 
                                      batched=True)

    print(f"Train sequences: {len(tokenized_train)}, Evaluation sequences: {len(tokenized_eval)}")

    # generate grid of parameters to find optimal combination
    param_grid = {"batch_size": params["batch_size"], 
                  "learning_rate": params["learning_rate"],
                  "epochs": params["epochs"]}

    # iterate over combinations of parameters
    for model_params in list(ParameterGrid(param_grid)):
    
        print(f"Training model configuration: {model_params}")

        # load the pre-trained model
        num_labels = 2
        model = AutoModelForSequenceClassification.from_pretrained(pretrain_path, num_labels=num_labels)

        # get current params
        learning_rate = float(model_params["learning_rate"])
        epochs = model_params["epochs"]
        batch_size = model_params["batch_size"]

        # consolidate training arguments
        training_args = TrainingArguments(output_dir=checkpoint_path,
                                        overwrite_output_dir=True,
                                        learning_rate=learning_rate,
                                        num_train_epochs=epochs,
                                        per_device_train_batch_size=batch_size,
                                        per_device_eval_batch_size=batch_size ,
                                        evaluation_strategy="no",
                                        save_strategy="no"
                                        )
                                        
        # by default the Trainer will use CrossEntropy loss from (torch.nn)
        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=nlp_utils.compute_metrics,
            train_dataset=tokenized_train, 
            eval_dataset=tokenized_eval
        )

        #--------------------
        # TRAINING
        #--------------------

        # train model
        trainer.train()

        # evaluate on full evaluation dataset
        results = trainer.predict(tokenized_eval)
        final_metrics = results[2]

        # save results
        df_temp = pd.DataFrame({"model": ["distilbert"],
                                "k_fold": [split_idx],
                                "max_seq_len": [max_sent_size],
                                "batch_size": [batch_size],
                                "epochs": [epochs],
                                "frozen_layers": 0,
                                "learning_rate": [learning_rate],
                                "num_validation": [len(tokenized_eval)],
                                "loss": [final_metrics["test_loss"]],
                                "accuracy": [final_metrics["test_accuracy"]],
                                "precision": [final_metrics["test_precision"]],
                                "recall": [final_metrics["test_recall"]],
                                "f1": [final_metrics["test_f1"]],
                                })

        # update global dataframe
        df_results = pd.concat([df_results, df_temp])

        # save dataframe with final results every iteration
        df_results.to_csv(output_path + "cv_results.csv", index=False)

        # print duration
        duration = (time.perf_counter() - start)/60
        print(f"Cross validation duration: {duration} minutes")

#%%

#=============================
# 4. Predictions on test set with optimal model
#=============================

# tokenize test set
dataset_test = Dataset.from_pandas(df_test)
full_test_dataset = dataset_test.map(nlp_utils.tokenize_function, 
                                      fn_kwargs={"tokenizer": tokenizer, "col_text": "sequence", "max_sent_size": max_sent_size}, 
                                      batched=True)

# tokenize all train data together (no validation split)
dataset_train = Dataset.from_pandas(df_train)
full_train_dataset = dataset_train.map(nlp_utils.tokenize_function, 
                                       fn_kwargs={"tokenizer": tokenizer, "col_text": "sequence", "max_sent_size": max_sent_size}, 
                                       batched=True)

#%%

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

# load the model
num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(pretrain_path, num_labels=num_labels)

# define arguments
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
    train_dataset=full_train_dataset, 
    eval_dataset=full_test_dataset
)

# train model
trainer.train()

#%%

# get predictions for full test data
results = trainer.predict(full_test_dataset)
final_metrics = results[2]

# save predictions on test set
logits = results.predictions
softmax = torch.nn.Softmax(dim=0)
probs = [softmax(torch.tensor(l)).numpy() for l in logits]
positive_probs = [prob[1] for prob in probs]
df_test["wfh_prob"] = positive_probs
df_test.to_csv(output_path + "test_data_predictions.csv", index=False)

# %%
