#%%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# paths
data_path = "../data/"
output_path = "../output/"
# %%

#=============
# Train data description
#=============

df = pd.read_csv(data_path + "train_sequences.csv")

# for each source of data calculate several metrics
df_results = pd.DataFrame()
for source in df["source"].unique():
    print(f"Calculating metrics for: {source}")
    df_temp = df.loc[df["source"] == source]
    
    # 1. all texts
    all_text = len(df_temp)
    # 2. number of unique sequences (according to their text)
    unique_text = len(df_temp.groupby("sequence").size())
    # 3a. tabulate number of labels per unique text
    display(df_temp.groupby("sequence", as_index=False).size()["size"].value_counts())
    # 3b. average number of labels per unique text
    avg_labels = df_temp.groupby("sequence").size().mean()
    # 4. agreement rate for unique texts
    df_grouped = df_temp.groupby("sequence").mean()
    num_agree = len(df_grouped.loc[(df_grouped["labels"] == 0) | (df_grouped["labels"] == 1)])
    perc_agree = num_agree/unique_text
    
    # save metrics
    temp_metrics = [int(all_text), unique_text, avg_labels, perc_agree]
    df_results[source] = temp_metrics


metrics_names = pd.Series(["all_seq", "unique_seq", "labels_per_seq", "agreement_rate"])
df_results.set_index([metrics_names], inplace=True)
df_results.to_csv(output_path + "train_description.csv")
np.round(df_results, 2)
#%%

#=============
# Test data description
#=============

df = pd.read_csv(data_path + "test_sequences.csv")

# for each source of data calculate several metrics
df_results = pd.DataFrame()
for source in df["source"].unique():
    print(f"Calculating metrics for: {source}")
    df_temp = df.loc[df["source"] == source]
    
    # 1. all texts
    all_text = len(df_temp)
    # 2. number of unique sequences (according to their text)
    unique_text = len(df_temp.groupby("sequence").size())
    # 3a. tabulate number of labels per unique text
    display(df_temp.groupby("sequence", as_index=False).size()["size"].value_counts())
    # 3b. average number of labels per unique text
    avg_labels = df_temp.groupby("sequence").size().mean()
    # 4. agreement rate for unique texts
    df_grouped = df_temp.groupby("sequence").mean()
    num_agree = len(df_grouped.loc[(df_grouped["labels"] == 0) | (df_grouped["labels"] == 1)])
    perc_agree = num_agree/unique_text
    
    # save metrics
    temp_metrics = [all_text, unique_text, avg_labels, perc_agree]
    df_results[source] = temp_metrics


metrics_names = pd.Series(["all_seq", "unique_seq", "labels_per_seq", "agreement_rate"])
df_results.set_index([metrics_names], inplace=True)
df_results.to_csv(output_path + "test_description.csv")
np.round(df_results, 2)

# %%

#=============
# Final AMT data description
#=============

# load label-level AMT data
df_amt = pd.read_csv(data_path + "final_amt_label_level.csv")
df_amt
# %%

# TABLE: Agreement rates between human AMT labelers
df_amt.groupby("seq_id").sum().groupby("actual_label", as_index=False).size()

# %%

#=============
# Test AMT data vs. WHAM
#=============

# read WHAM predictions and produce a binary classification
threshold = 0.5

df_wham = pd.read_csv(data_path + "v1_test_sequences_predictions.csv")
df_wham["wfh_bert"] = df_wham["wfh_prob"].apply(lambda x: 1 if x >= threshold else 0)

# groupb by sequence and create majority label
df_seq_level = df_wham.groupby("sequence_id", as_index=False).mean()
df_seq_level["majority"] = df_seq_level["original_labels"].apply(lambda x: round(x))
df_seq_level
#%%

def generate_cm(df, label_col, compare_col, name, normalize=False):

    accuracy = accuracy_score(df[label_col], df[compare_col])
    print(f"{name} accuracy: {accuracy}")
    cm = confusion_matrix(df[label_col], df[compare_col], normalize=normalize)
    cm = np.round(cm, 3)

    # set size of font for all plots
    plt.rcParams["font.family"] = "serif"
    sns.set(font_scale=1.5)

    plt.figure(figsize=(12,10))
    ax = sns.heatmap(cm, annot=True, fmt='g',
                    cbar=False, square=True, 
                    cmap = "Blues",
                    #cmap=ListedColormap(['white']),
                    linewidths=0.5)

    ax.set_title(f'{name} confusion matrix \n ({len(df)} labeled sequences, accuracy: {np.round(accuracy, 2)}) \n', fontsize=22)
    ax.set_xlabel(f'\n{name}', weight="bold", fontsize=18)
    ax.set_ylabel('Human labels \n\n', weight="bold", fontsize=18)

    # label ticks
    ax.xaxis.set_ticklabels(['WFH = 0','WFH = 1'])
    ax.yaxis.set_ticklabels(['WFH = 0','WFH = 1'])

    #plt.savefig(plots_path + "cm_bert.pdf")
    plt.show()

    return cm


# TABLE: Agreement between human labels and WHAM predictions
cm = generate_cm(df_seq_level, "majority", "wfh_bert", "WHAM", None)

# generate a normalized version
cm_norm = generate_cm(df_seq_level, "majority", "wfh_bert", "WHAM", "all")

#%%

# DATA POINTS: Breakdown of confusion matrix according to agreement
cm_agree = generate_cm(df_seq_level.loc[df_seq_level["disagree"] == 0], "majority", "wfh_bert", "WHAM Agreement Seq", "all")
cm_disagree = generate_cm(df_seq_level.loc[df_seq_level["disagree"] == 1], "majority", "wfh_bert", "WHAM Disagreement Seq", "all")
# %%

#=============
# Test AMT data vs. Dictionaries
#=============

df_dict = pd.read_csv(data_path +  "dict_replication_test_sequences.csv")
df_dict = df_dict[["sequence_id.x", "narrow_replication", "neg_dict_replication"]]
df_dict.columns = ["sequence_id", "narrow_result", "neg_narrow_result"]
df_dict

#%%

# join with AMT labels
df_seq_level = pd.merge(df_seq_level, df_dict, how="inner", on="sequence_id")
df_seq_level

# %%

# TABLE: Agreement between human labels and dictionary
cm_dict = generate_cm(df_seq_level, "majority", "narrow_result", "Dictionary", None)
cm_dict_norm = generate_cm(df_seq_level, "majority", "narrow_result", "Dictionary", "all")

# %%

# TABLE: Agreement between human labels and dictionary
cm_dict_neg = generate_cm(df_seq_level, "majority", "neg_narrow_result", "Dictionary with negation", None)
cm_dict_neg_norm = generate_cm(df_seq_level, "majority", "neg_narrow_result", "Dictionary with negation", "all")

# %%
