# EcoRank
This repository contains the source code for the paper `EcoRank: Budget-Constrained Text Re-ranking Using Large Language Models`

## Setup
We use python 3.9 for our experiments. Install the necessary libraries from the `requirements.txt` file. Minimum GPU needed to load Flan T5-XL model (AWS EC2 g5.4xlarge instance used for our experiments)

## Getting the dataset and evidence passages
We use the test split of the NQ and WQ datasets. For evidence collection, we use wikipedia 2018 English dump. The code to collect the datasets and wikipedia split is taken from [UPR](https://github.com/DevSinghSachan/unsupervised-passage-reranking) code repository. To collect the evidence passages as a `tsv` file run the following command:
```sh
python3 download_data.py --resource data.wikipedia-split.psgs_w100
```
To download the NQ-test dataset run:
```sh
python3 download_data.py --resource data.retriever-outputs.bm25.nq-test
```
To download the WQ-test dataset run:
```sh
python3 download_data.py --resource data.retriever-outputs.bm25.webq-test
```
These will download the passages into `./downloads/data/wikipedia-split/` and datasets into `./downloads/data/retriever-outputs/bm25/` folders.

### Process the Wikipedia passages
Convert the Wikipedia passages into a dictionary where the key is the id of the passage and value is the text of the passage:
```sh
python3 process_wikipedia.py
```
This will create a pickle file `wiki_id2text.pickle` in the home directory.

## Run EcoRank
```sh
python3 run_ecorank.py --input_dataset ./downloads/data/retriever-outputs/bm25/nq-test.json
```

You can change different parameters of EcoRank such as budget split, budget, etc. within the code file.
