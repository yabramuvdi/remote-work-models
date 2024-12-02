"""
Test a language model on the task of masked language modeling (MLM). The user can
interact with the script through the terminal by providing any sequence of text with 
a masked word (replacing the word with [MASK]). The pre-trained language model will
be used to "unmask" the word. The user can determine the number of predicted words
to get from the model.

Example command on the terminal:

    python test_model.py "This job can be done from the comfort of your [MASK]." 10

"""

if __name__ == "__main__":

    from transformers import AutoModelForMaskedLM, AutoTokenizer, FillMaskPipeline
    import torch
    import sys
    import numpy as np

    # unpack command line arguments
    input_text = str(sys.argv[1])
    top_k = int(sys.argv[2])

    # select the appropriate device for computations
    if torch.cuda.is_available():
        device = 0
    else:
        device = -1

    final_model_path = "../models/pre-train/final/"
    #final_model_path = "distilbert-base-uncased"

    # load the pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(final_model_path, padding=True)
    model = AutoModelForMaskedLM.from_pretrained(final_model_path) 

    # create a pipeline for unmasking hidden words
    unmasker = FillMaskPipeline(model=model,
                                tokenizer=tokenizer,
                                device=device,
                                top_k=top_k)

    # transform the input text into the appropriate format
    test_text = input_text.replace("[MASK]", f"{unmasker.tokenizer.mask_token}")
    
    #test_text = f"This position is a {unmasker.tokenizer.mask_token} position."
    #test_text = f"You will mainly work from {unmasker.tokenizer.mask_token} in this position."

    # get predictions
    output = unmasker(test_text)

    # print results
    print(f"\nTop {top_k} predictions for the missing word in the the text: {test_text} \n")
    print("=========")
    for i, result in enumerate(output, start=1):
        print(f"{i}. {result['token_str']} (with probability: {np.round(result['score'], 3)})")

    print("=========")