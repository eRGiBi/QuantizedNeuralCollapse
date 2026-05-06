"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save shak_char_train.bin, shak_char_val.bin containing the ids, and shak_char_meta.pkl containing the
encoder and decoder and some other related info.

https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/prepare.py
"""


import os
import pickle
import requests
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


# # download the tiny shakespeare dataset
# input_file_path = os.path.join('data', 'shakes_char_input.txt')
#
# if not os.path.exists(input_file_path):
#     data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
#     with open(input_file_path, 'w') as f:
#         f.write(requests.get(data_url).text)
#
# with open(input_file_path, 'r') as f:
#     data = f.read()
# print(f"length of dataset in characters: {len(data):,}")
#
# # get all the unique characters that occur in this text
# chars = sorted(list(set(data)))
# vocab_size = len(chars)
# print("all the unique characters:", ''.join(chars))
# print(f"vocab size: {vocab_size:,}")
#
# # create a mapping from characters to integers
# stoi = { ch:i for i,ch in enumerate(chars) }
# itos = { i:ch for i,ch in enumerate(chars) }
# def encode(s):
#     return [stoi[c] for c in s] # encoder: take a string, output a list of integers
# def decode(l):
#     return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
#
# # create the train and test splits
# n = len(data)
# train_data = data[:int(n*0.9)]
# val_data = data[int(n*0.9):]
#
# # encode both to integers
# train_ids = encode(train_data)
# val_ids = encode(val_data)
# print(f"train has {len(train_ids):,} tokens")
# print(f"val has {len(val_ids):,} tokens")
#
# # export to bin files
# train_ids = np.array(train_ids, dtype=np.uint16)
# val_ids = np.array(val_ids, dtype=np.uint16)
# train_ids.tofile(os.path.join(os.path.dirname(__file__), 'shak_char_train.bin'))
# val_ids.tofile(os.path.join(os.path.dirname(__file__), 'shak_char_val.bin'))
#
# # save the meta information as well, to help us encode/decode later
# meta = {
#     'vocab_size': vocab_size,
#     'itos': itos,
#     'stoi': stoi,
# }
# with open(os.path.join(os.path.dirname(__file__), 'shak_char_meta.pkl'), 'wb') as f:
#     pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens


class ShakespeareCharDataset(IterableDataset):
    """character-level dataset using memory-mapped files
    """

    def __init__(self, data_path, block_size=1024, split_size=None):
        self.data_path = data_path
        self.block_size = block_size
        self.split_size = split_size

        # Get data length without keeping memmap in memory
        data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.data_len = len(data)
        del data

        self.n_samples = self.data_len - self.block_size - 1

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        """Yield random samples from the dataset."""
        count = 0
        while True:
            if self.split_size is not None and count >= self.split_size:
                break

            # Recreate memmap per sample to avoid memory leak
            # data = np.memmap(self.data_path, dtype=np.float32, mode="r")
            data = np.memmap(self.data_path, dtype=np.uint16, mode="r")

            # Random start index
            idx = torch.randint(0, self.data_len - self.block_size - 1, (1,)).item()

            # Extract sequence
            x = torch.from_numpy(data[idx: idx + self.block_size].astype(np.int64)).clone()
            y = torch.from_numpy(data[idx + 1: idx + self.block_size + 1].astype(np.int64)).clone()

            del data  # Release memmap

            yield x, y
            count += 1


class ShakespeareCharDatasetIndexed(Dataset):
    """Indexed version for analysis/validation for deterministic ordering."""

    def __init__(self, data_path, block_size=1024, max_samples=None):
        self.data_path = data_path
        self.block_size = block_size

        data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.data_len = len(data)
        del data

        self.n_samples = self.data_len - self.block_size - 1
        if max_samples is not None:
            self.n_samples = min(self.n_samples, max_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """Get a specific sequence by index."""
        # Recreate memmap per access
        data = np.memmap(self.data_path, dtype=np.uint16, mode="r")

        x = torch.from_numpy(data[idx : idx + self.block_size].astype(np.int64)).clone()
        y = torch.from_numpy(data[idx + 1 : idx + self.block_size + 1].astype(np.int64)).clone()

        del data

        return x, y


def prepare_shakespeare_char_dataset(
        data_dir="data/shakespeare_char",
        block_size=1024,
        batch_size=12,
        ood_filename="shak_char_ood.bin",
        train_split_size: int = 10000,
        deterministic_train: bool = False,
):
    """Prepare Shakespeare character dataset with train and analysis splits """
    train_path = os.path.join(data_dir, "shak_char_train.bin")
    val_path = os.path.join(data_dir, "shak_char_val.bin")
    meta_path = os.path.join(data_dir, "shak_char_meta.pkl")
    ood_path = os.path.join(data_dir, "shak_char_ood.bin")

    vocab_size = 65

    if os.path.exists(meta_path):
        import pickle

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        vocab_size = meta.get("vocab_size", 65)

    # Training set:
    # - random sampling (IterableDataset) is fine for general LM training but makes "train accuracy" noisy
    # - deterministic indexed dataset is preferable for overfitting/terminal-phase NC measurements
    if deterministic_train:
        trainset = ShakespeareCharDatasetIndexed(
            train_path, block_size=block_size, max_samples=train_split_size
        )
    else:
        trainset = ShakespeareCharDataset(
            train_path, block_size=block_size, split_size=train_split_size
        )

    # # Analysis set: Indexed dataset for deterministic evaluation
    # analysis_set = ShakespeareCharDatasetIndexed(
    #     val_path, block_size=block_size, max_samples=1000
    # )

    # Validation: Deterministic, check standard generalization
    val_dataset = ShakespeareCharDatasetIndexed(
        val_path, block_size=block_size, max_samples=500
    )

    # OOD: Deterministic, check behavior on distribution shift
    if os.path.exists(ood_path):
        ood_dataset = ShakespeareCharDatasetIndexed(
            ood_path, block_size=block_size, max_samples=500
        )
    else:
        print(f"Warning: OOD file {ood_path} not found. Using Val set as placeholder.")
        ood_dataset = val_dataset

    return trainset, val_dataset, ood_dataset, vocab_size
