import os
import requests

import torch
from torch.utils.data import Dataset

from model_architectures.shakespeare_char_model import batch_size


class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def prepare_text_char_dataset(
        config,
        reduced=False
):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    if reduced:
        file_path = "data/shakespeare_all/reduced_input.txt"
    else:
        file_path = "data/shakespeare_all/input.txt"

    if not os.path.exists(file_path):
        print("Downloading")
        with open(file_path, "w") as f:
            f.write(requests.get(url).text)

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    config["vocab_size"] = vocab_size
    config["num_classes"] = vocab_size
    print(f"Inferred vocab_size={vocab_size} from dataset")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Encoder and Decoder functions
    encode = lambda s: [stoi[c] for c in s]

    # Process data into tensors
    data = torch.tensor(encode(text), dtype=torch.long)

    # Split into Train and Val, test
    split = config.get("train_val_test_split", [0.8, 0.1, 0.1])
    n1 = int(split[0] * len(data))
    n2 = int((split[0] + split[1]) * len(data))
    
    train_data = data[:n1]
    val_data = data[n1:n2]
    ood_data = data[n2:]

    block_size = config.get("block_size")

    def get_batch(split_data):
        # Generate random starting indices for the batch
        ix = torch.randint(len(split_data) - block_size, (config.get("batch_size"),))
        x = torch.stack([split_data[i:i + block_size] for i in ix])
        y = torch.stack([split_data[i + 1:i + block_size + 1] for i in ix])
        return x, y

    return (
        CharDataset(train_data, block_size),
        CharDataset(val_data, block_size),
        CharDataset(ood_data, block_size),
        vocab_size
    )
    # return train_data, val_data, ood_data, vocab_size
