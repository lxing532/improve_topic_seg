# Improving Neural Topic Segmentation with Coherence-Related Auxiliary Task and Restricted Self-Attention
This repository maintains the source code for "[Improving Context Modeling in Neural Topic Segmentation](https://arxiv.org/pdf/2010.03138.pdf)", AACL-IJCNLP 2020.

## The codes for basic neural topic segmenter
Please note the main framework of our method is mostly adopted from [Text Segmentation as a Supervised Learning Task](https://arxiv.org/abs/1803.09337). In order to protect the copyright of this previously proposed topic segmenter implemented by Koshorek et al., we only upload the files which were modified to align with our work. Hence, if you consider running the code, please follow [this link](https://github.com/koomri/text-segmentation) to the original repository and build virtual environment and download the necessary packages as required. Afterwards, please replace the files in [this link](https://github.com/koomri/text-segmentation) with the files in our repo.

## run.py, wiki_loader.py and choiloader.py
Please replace the these three files located at the root directory of the original repo with the ones provided in our repo.

## max_sentence_embedding.py
This file is originally under the "models" folder, please replace the one in the original repo with the one provided by us.

## BERT sentence embedding
We use [bert-as-service](https://github.com/hanxiao/bert-as-service) to first generate the BERT sentence embeddings for each document, then we store all the documents' sentence embeddings into a .txt file. In order to use our code, you should also generate the BERT file first and then load it in run.py.

To describe the structure of the BERT .txt file more clearly, let's use an example:

* Let's say we have a tiny corpus with two documents, each document has 5 sentences. 

* We first generate the BERT sentence embeddings for each document. Thus, we have two sets of embeddings, each set contains 5 sentence embeddings, with the dimension 768.

* Then, we put these two sets of embeddings into a .txt file, each embedding takes one line. We add a blank line between two documents. Thus, this .txt file will have 11 lines (5+1+5). 
