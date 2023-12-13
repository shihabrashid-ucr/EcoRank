# EcoRank

## Getting the dataset and evidence passages
We use the test split of the NQ and WQ datasets. For evidence collection we use wikipedia 2018 English dump. The code to collect the datasets and wikipedia split is taken from UPR[https://github.com/DevSinghSachan/unsupervised-passage-reranking] code repository. To collect the evidence passages as a `tsv` file run the following command:
```sh
python download_data.py --resource data.wikipedia-split.psgs_w100
```
To download the NQ-test dataset run:
```sh
python download_data.py --resource data.retriever-outputs.bm25.nq-test
```
These will download the passages into `./downloads/data/wikipedia-split/` and datasets into `./downloads/data/retriever-outputs/bm25/` folders. 
