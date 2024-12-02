#%%

import dict_methods_functions as dm
import string
import pandas as pd

input_path = "../input/"
# %%

#### 1. Test the text cleaning function

# define punctuation symbols to remove
punctuation = string.punctuation
punctuation = punctuation.replace("-", "")
punctuation = punctuation.replace("'", "")

input_seq = "This a simple sentece with some numbers 1234 and some punctuation symbols !@#$%^&*()_+ and CAPITAL letters."
print(input_seq)
print("========================")
print(dm.clean_sequence(input_seq, punctuation))

#%%

#### 2. Test the function that generates a regular expression from multi-word terms
dm.gen_multiple_word_regex("work from home")

#%%

# load dictionary
dict_df = pd.read_csv(input_path + "dictionary.csv")
dict_list = dict_df["term"].tolist()

# generate a regular expression from the dictionary
dict_regex = dm.gen_regex_from_dict(dict_list)
dict_regex
#%%

test_cases = ["this is a home-based job.",
              "we are looking for a home based worker.",
              "we can remotely work.",
              "home- based position to work from home"]

for case in test_cases:
    print("Test case: " + case)
    results = dm.find_regex_matches(dict_regex, case)
    print("Found matches: " + str(results))
# %%

# apply on a dataframe
df_test = pd.DataFrame({"sequence": test_cases}) 
df_test["sequence_clean"] = df_test["sequence"].apply(lambda x: dm.clean_sequence(x, punctuation))
tuples_matches = df_test.apply(lambda row: dm.apply_find_regex_matches(row, "sequence_clean", dict_regex), axis=1)
df_test[['dict_matches', 'dict_positions']] = pd.DataFrame(tuples_matches.tolist(), index=df_test.index)
df_test

#%%

# print some random examples (sequences with matches)
for i, row in df_test[df_test["dict_matches"].apply(lambda x: len(x) > 0)].sample(1).iterrows():
    #print("Sequence: " + row["sequence.x"])
    print("Clean sequence: " + row["sequence_clean"])
    print("Matches: " + str(row["dict_matches"]))
    print("Positions: " + str(row["dict_positions"]))
    print("=================================\n")

# %%

#### 3. Test the functions that identify negations

tokenization_pattern = r'''
                        (?x)                # set flag to allow verbose regexps (to separate logical sections of pattern and add comments)
                        \w+(?:-\w+)*        # word characters with internal hyphens
                        \w+(?:'\w+)*        # word characters with internal apostrophes
                        |\b\w+\b            # single letter words
                        '''

# load negation terms
neg_df = pd.read_csv(input_path + "negation.csv")

# parameters
neg_terms_before = list(neg_df["term"])
neg_terms_after = ["no", "not"]
window_before = 3
window_after = 2
search_nt = True
text_col = "sequence_clean"

# apply binary tagging
df_test["negation"] = df_test.apply(dm.find_negation, args=(text_col, neg_terms_before, neg_terms_after, window_before, window_after, search_nt, tokenization_pattern), axis=1)
df_test
# %%

# print some random examples (sequences with matches and negation)
for i, row in df_test[(df_test["dict_matches"].apply(lambda x: len(x) > 0)) & (df_test["negation"] == 1)].sample(5).iterrows():
    #print("Sequence: " + row["sequence.x"])
    print("Clean sequence: " + row["sequence_clean"])
    print("Matches: " + str(row["dict_matches"]))
    print("Positions: " + str(row["dict_positions"]))
    print("Negation: " + str(row["negation"]))
    print("=================================\n")
# %%

# apply tagging with negation terms
neg_results = df_test.apply(dm.find_negation_with_terms, args=(text_col, neg_terms_before, neg_terms_after, window_before, window_after, search_nt, tokenization_pattern), axis=1)
neg_binary, neg_terms = list(zip(*neg_results))
df_test["negation"] = neg_binary
df_test["negation_terms"] = neg_terms
df_test

#%%

# print some random examples (sequences with matches and negation)
for i, row in df_test[(df_test["dict_matches"].apply(lambda x: len(x) > 0)) & (df_test["negation"] == 1)].sample(5).iterrows():
    #print("Sequence: " + row["sequence.x"])
    print("Clean sequence: " + row["sequence_clean"])
    print("Matches: " + str(row["dict_matches"]))
    print("Positions: " + str(row["dict_positions"]))
    print("Negation: " + str(row["negation"]))
    print("Negation terms: " + str(row["negation_terms"]))
    print("=================================\n")
# %%

#### 4. Test the function that identifies generic terms

# load generic terms
df_generic = pd.read_csv(input_path + "generic.csv")
df_generic

# generate a regular expression for generic terms
generic_regex  = ""
for _, row in df_generic.iterrows():
    term = row["term"]
    part = row["part of word"]

    if part == "no":
        # add word boundaries to the regex
        term = r"\b" + term + r"\b"
        generic_regex += term + r"|"
    elif part == "yes":
        # add only starting word boundary to the regex
        term = r"\b" + term
        generic_regex += term + r"|"

# clean last pipe
generic_regex = generic_regex[:-1]
generic_regex

# %%
