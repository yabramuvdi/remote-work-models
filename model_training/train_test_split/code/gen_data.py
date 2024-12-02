#================
# Generate train and test data with only sequences labeled on AMT
#================

#%%

import pandas as pd
data_path = "../data/"

#%%

#====
# Core AMT labels
#====

# load all data from the final round of AMT labeling
df_complete = pd.read_csv(data_path + "final_amt_label_level.csv")
df_complete

#%%

# load all test data and select only the one from final AMT
df_test = pd.read_csv(data_path + "test_complete_sequences.csv")
df_test_core = df_test.loc[df_test["source"] == "final_amt"]
df_test_core

#%%

# generate training data as the remaining examples
df_train_core = df_complete.loc[~df_complete["seq_id"].isin(df_test["sequence_id"])]
df_train_core = df_train_core[["seq_id", "country", "part", "sequence", "actual_label"]]
df_train_core.columns = ["sequence_id", "country", "part", "sequence", "label"]
df_train_core.to_csv(data_path + "train_core_amt.csv", index=False)
df_train_core
# %%

# get additional information for test sequences
df_test_core = pd.merge(df_test_core, df_complete[["seq_id", "country", "part"]], how="left", left_on="sequence_id", right_on="seq_id")
df_test_core = df_test_core[["sequence_id", "country", "part", "sequence", "original_labels"]]
df_test_core.columns = ["sequence_id", "country", "part", "sequence", "label"]
df_test_core = df_test_core.groupby("sequence_id", as_index=False).max()
df_test_core.to_csv(data_path + "test_core_amt.csv", index=False)
df_test_core
# %%

#====
# Additional training labels
#====

df_train_all = pd.read_csv(data_path + "train_complete_sequences.csv")
df_train_all

#%%

df_train_not_core = df_train_all.loc[~df_train_all["sequence_id"].isin(df_train_core["sequence_id"])]
df_train_not_core = df_train_not_core[["sequence_id", "sequence", "labels"]]
df_train_not_core.columns = ["sequence_id", "sequence", "label"]
df_train_not_core

#%%

df_train_not_core.to_csv(data_path + "train_supplementary.csv", index=False)
df_train_not_core
# %%
