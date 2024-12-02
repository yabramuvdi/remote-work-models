from datasets import load_metric
import numpy as np

def tokenize_function(df_examples, tokenizer, col_text, max_sent_size=512, padding="max_length", truncation=True):
    """ Function for tokenizing the text of individual sequences of text. 
        Here we define the main parameters for text tokenization.

    df_examples: Pandas dataframe
        Main dataset with the text data

    tokenizer: HuggingFace Tokenizer
        Tokenizer object

    col_text: str
        Name of the column that contains the text data
    """

    return tokenizer(
        df_examples[col_text],
        padding=padding,
        truncation=truncation,
        max_length=max_sent_size,
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask`.
        return_special_tokens_mask=True,
    )


# available metrics: https://huggingface.co/metrics
def compute_metrics(eval_pred):
    metric1 = load_metric("precision")
    metric2 = load_metric("recall")
    metric3 = load_metric("f1")
    metric4 = load_metric("accuracy")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    precision = metric1.compute(predictions=predictions, references=labels)["precision"]
    recall = metric2.compute(predictions=predictions, references=labels)["recall"]
    f1 = metric3.compute(predictions=predictions, references=labels)["f1"]
    accuracy = metric4.compute(predictions=predictions, references=labels)["accuracy"]
    
    return {"precision": precision, "recall": recall, 
            "f1": f1, "accuracy": accuracy}

