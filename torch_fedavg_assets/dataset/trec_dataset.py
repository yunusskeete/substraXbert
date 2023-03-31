import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["TOKENIZERS_PARALLELISM"] = "true"


def setup_trec(data_path, N_CLIENTS):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Download the dataset
    trec_train = load_dataset("trec", split="train[:1000]")
    trec_test = load_dataset("trec", split="test[:100]")

    # tokenize everything at once
    train_tokens = tokenizer(
        trec_train['text'], max_length=512,
        truncation=True, padding='max_length'
    )
    test_tokens = tokenizer(
        trec_test['text'], max_length=512,
        truncation=True, padding='max_length'
    )

    # initialize arrays to be used
    train_labels = np.zeros(
        (len(trec_train),
         max(trec_train['coarse_label'])+1),
         dtype="float32"
    )
    test_labels = np.zeros(
        (len(trec_test),
         max(trec_test['coarse_label'])+1),
         dtype="float32"
    )
    # one-hot encode
    train_labels[np.arange(len(trec_train)), trec_train['coarse_label']] = 1
    test_labels[np.arange(len(trec_test)), trec_test['coarse_label']] = 1

    # Split arrays into the number of organizations
    train_ids_folds = np.split(np.array(train_tokens['input_ids']), N_CLIENTS)
    train_attention_mask_folds = np.split(np.array(train_tokens['attention_mask']), N_CLIENTS)
    train_labels_folds = np.split(train_labels, N_CLIENTS)
    test_ids_folds = np.split(np.array(test_tokens['input_ids']), N_CLIENTS)
    test_attention_mask_folds = np.split(np.array(test_tokens['attention_mask']), N_CLIENTS)
    test_labels_folds = np.split(test_labels, N_CLIENTS)
    print(f"Number of splits: {len(test_labels_folds)}")
    print(f"Length of train dataset: {len(train_labels)}")
    print(f"Length of train splits: {len(train_labels_folds[0])}")
    print(f"Length of test dataset: {len(test_labels)}")
    print(f"Length of test splits: {len(test_labels_folds[0])}")

    # Save splits in different folders to simulate the different organizations
    for i in range(N_CLIENTS):

        # Save train dataset on each org
        os.makedirs(os.path.join(data_path, f"org_{i+1}/train"), exist_ok=True)

        filename = os.path.join(data_path, f"org_{i+1}/train/train_ids.npy")
        np.save(str(filename), train_ids_folds[i])
        filename = os.path.join(data_path, f"org_{i+1}/train/train_attention_mask.npy")
        np.save(str(filename), train_attention_mask_folds[i])
        filename = os.path.join(data_path, f"org_{i+1}/train/train_labels.npy")
        np.save(str(filename), train_labels_folds[i])


        # Save test dataset on each org
        os.makedirs(os.path.join(data_path, f"org_{i+1}/test"), exist_ok=True)

        filename = os.path.join(data_path, f"org_{i+1}/test/test_ids.npy")
        np.save(str(filename), test_ids_folds[i])
        filename = os.path.join(data_path, f"org_{i+1}/test/test_attention_mask.npy")
        np.save(str(filename), test_attention_mask_folds[i])
        filename = os.path.join(data_path, f"org_{i+1}/test/test_labels.npy")
        np.save(str(filename), test_labels_folds[i])
