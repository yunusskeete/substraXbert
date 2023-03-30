# Trec

This dataset is the [Text REtrieval Conference (TREC) Question Classification dataset](https://huggingface.co/datasets/trec).
It is download from Hugging Face.

The dataset has 6 coarse class labels and 50 fine class labels.
Average length of each sentence is 10, vocabulary size of 8700.

## Data repartition

### Train and test

### Split data between organizations

## Opener usage

The opener exposes 2 methods:

- `get_data` returns a dictionary containing the text as a string and the coarse label and the fine label as integers
- `fake_data` returns a fake string, coarse label and fine label in a dict
