import pandas as pd
import numpy as np
import re
import string
import nltk

#=================
# Preprocessing steps
#=================

def remove_punctuation(sequence, punctuation):
    # remove punctuation symbols and add a whitespace instead         
    regex = re.compile('[%s]' % re.escape(punctuation))
    clean_seq = regex.sub(' ', sequence)
    return clean_seq

# apply preprocessing steps to a sequence
def clean_sequence(seq, punctuation):
    # lowercase
    clean_seq = seq.lower()
    # remove punctuation
    #clean_seq = remove_punctuation(clean_seq, punctuation)
    # remove all numbers
    #clean_seq =  ''.join([i for i in clean_seq if not i.isdigit()])
    # remove extra white spaces
    #clean_seq = re.sub(r"\s+", " ", clean_seq)

    return clean_seq

#=================
# Generate regular expressions from dictionaries
#=================

def gen_multiple_word_regex(text):
    # craft a regular expression that matches any combination of hyphens 
    # and white spaces separating the words in a dictionary term
    
    components = text.split(" ")
    term_regex = r"\b"
    for i, c in enumerate(components):
        if i < len(components)-1:
            term_regex += c + r"[- ]+"
        else:
            term_regex += c + r"\b"
        
    return term_regex

# generate a complete regular expression using all terms in the dictionary
# we will use word boundaries to delimit dictionary terms
def gen_regex_from_dict(dict_list):
    
    # we start by sorting the dictionary terms by number of unigrams (descending)
    # this is important since we will match regular expressions in order
    dict_list.sort(key=lambda x: (len(x.split(" ")), len(x)), reverse=True)
    print("Dictionary terms sorted by length (descending)")

    dict_regex = ""
    for term in dict_list:
        if len(term.split(" ")) == 1:
            term = r"\b" + term + r"\b"
            dict_regex += term + r"|"
        else:
            dict_regex += gen_multiple_word_regex(term) + r"|"
            
    dict_regex = dict_regex[:-1]
    return dict_regex

# use a regex to find all the matches that each sequence has.
# capture both the expression/s that generated the match and their positions
def find_regex_matches(regex, text):
    
    #pattern = re.compile(regex, re.IGNORECASE)
    matches = regex.finditer(text)
    results = [(match.group(0), (match.start(), match.end())) for match in matches]
    
    return results

# apply the function to a dataframe
def apply_find_regex_matches(text, regex):
    matches = find_regex_matches(regex, text)
    match_terms = [match[0] for match in matches]
    match_positions = [match[1] for match in matches]
    return match_terms, match_positions

#=================
# Identify negations
#=================

# we start by defining a tokenization strategy
def tokenize_text(text, pattern):
    # tokenize all documents
    return nltk.regexp_tokenize(text, pattern=pattern)

def clean_term(term):
    # generate standardized version of multi-word terms
    term_clean = term.replace("-", " ")
    term_clean = re.sub(r"\s+", " ", term_clean)
    return term_clean


# we will look for negation terms only K words before the dictionary matches and J words after
def find_negation(row,
                  text_col,
                  neg_terms_before, 
                  neg_terms_after, 
                  window_before, 
                  window_after,
                  search_nt, 
                  tokenization_pattern):
    
    # unpack relevant elements from dataframe
    terms = row["dict_matches"]
    positions = row["dict_positions"]
    text = row[text_col]

    # create a global indicator of negation
    negation = False

    # for each regex match we will search for negation terms before and after
    for term, position in zip(terms, positions):

        # search behind
        behind_text = text[0:position[0]]
        tokens_behind = tokenize_text(behind_text, tokenization_pattern)
        behind_window = tokens_behind[-window_before:]
        neg_behind = [w for w in behind_window if w in neg_terms_before]
        
        if neg_behind:
            # we don't need to search for negation terms after the match
            negation = True
            return negation

        else:
            # search for terms after the match
            after_text = text[position[1]:]
            tokens_after = tokenize_text(after_text, tokenization_pattern)
            after_window = tokens_after[:window_after]
            neg_after = [w for w in after_window if w in neg_terms_after]
            
            if neg_after:
                negation = True
                return negation
            
            # search if the word after the match contains n't
            elif search_nt and len(after_window) > 0:
                after_word = after_window[0]
                if "n't" in after_word:
                    negation = True
                    return negation
            
    return negation


# variation of the function above that, for each negated, term includes the used negation term
def find_negation_with_terms(row,
                             text_col,
                             neg_terms_before, 
                             neg_terms_after, 
                             window_before, 
                             window_after,
                             search_nt, 
                             tokenization_pattern):
    
    # unpack relevant elements from dataframe
    terms = row["dict_matches"]
    positions = row["dict_positions"]
    text = row[text_col]

    # clean terms in order to standardize matches
    terms = [clean_term(t) for t in terms]

    # create global objects
    #negated_terms = {t: False for t in terms}
    #negated_terms = {} 
    negated_terms = []
    negation = False

    # for each regex match we will search for negation terms before and after
    for i, (term, position) in enumerate(zip(terms, positions)):

        # search behind
        behind_text = text[0:position[0]]
        tokens_behind = tokenize_text(behind_text, tokenization_pattern)
        behind_window = tokens_behind[-window_before:]
        neg_behind = [w for w in behind_window if w in neg_terms_before]
        
        if neg_behind:
            negation = True
            #negated_terms[term] = neg_behind
            negated_terms.append((term, i, neg_behind[0]))
            continue

        # search for terms after the match
        after_text = text[position[1]:]
        tokens_after = tokenize_text(after_text, tokenization_pattern)
        after_window = tokens_after[:window_after]
        neg_after = [w for w in after_window if w in neg_terms_after]
        
        if neg_after:
            negation = True
            #negated_terms[term] = neg_after
            negated_terms.append((term, i, neg_after[0]))
            continue
        
        # search if the word after the match contains n't
        if search_nt and len(after_window) > 0:
            after_word = after_window[0]
            if "n't" in after_word:
                negation = True
                #negated_terms[term] = ["n't"]
                negated_terms.append((term, i, "n't"))

    return (negation, negated_terms)

#=================
# Generate features
#=================

# function to create additional columns in the dataframe signaling the presence of each term

# # create and populate one column for each term from the dictionary
# # we will use this later in our logistic regressions
# for term in dict_narrow_df:
#     if len(term.split(" ")) == 1:
#         seq_df[term] = seq_df["dict_matches"].apply(lambda x: term in x)
#     else:
#         components = term.split(" ")
#         term_regex = r"\b"
#         for i, c in enumerate(components):
#             if i < len(components)-1:
#                 term_regex += c + r"[- ]+"
#             else:
#                 term_regex += c
        
#         compiled_regex = re.compile(term_regex)
#         seq_df[term] = seq_df["dict_matches"].apply(lambda x: any((match := compiled_regex.search(item)) for item in x))
