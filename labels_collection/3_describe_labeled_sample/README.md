# Data description

1. [final_amt_label_level.csv](./data/final_amt_label_level.csv): Label-level data containing all the 30,000 labels we collected from the 10,000 unique sequences sent to Amazon Mechanical Turk.
2. [train_sequences.csv](./data/train_sequences.csv): Exact file on which BERT is finetuned to produce WHAM. Contains sequences from a variety of sources (e.g. internal team labels, AMT labels)
3. [test_sequences.csv](./data/test_sequences.csv): Sequences we use for testing the model. For the paper we subset this file to focus only on the 4,050 sequences that we obtained from the final AMT labeling effort.
4. [v1_test_sequences_predictions.csv](./data/v1_test_sequences_predictions.csv): Contains the WHAM predictions for each one of the sequences that are part of the test data. These sequences are unseen to the model.
5. [dict_replication_test_sequences.csv](./data/dict_replication_test_sequences.csv): Contains the dictionary tagging of all the test sequences.