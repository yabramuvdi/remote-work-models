# Remote work prediction language models

This repository contains the code for training the language models used in the paper [_“Remote Work across Jobs, Companies, and Space” (Hansen, Lambert, Bloom, Davis, Sadun & Taska, 2023)_](https://wfhmap.com/) to predict the spread of work from home. It is developed and maintained by [Yabra Muvdi](https://yabramuvdi.github.io). Unfortunately, the labeled data used for training is not yet available. The repository, however, provides clarity on the training procedure and can serve as a reference for other projects.


## Labels collection
 
 Given the enormous size of the job posting corpus, we had to create a clear strategy for sampling the subser of examples we wanted to label. Picking examples totally at random was an option but, given that remote work was not very prevalent in job postings in 2021, we faced the risk of ending up with a sample that contained only few positive examples (remote work allowed). To alleviate this, we crafted several dictionaries that captured ways in which remote work could be expressed in job postings and used them to sample part of the data.

 ## Model training

 The code in this task performs:
 
 1. Additional pre-training of a distilBert model
 2. Finetuning of a distilBert model using the labels collected. Optimal parameters for finetuning are choosen through K-fold cross-validation. 