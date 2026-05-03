import torch
import requests
import os
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def prepare_text_char_dataset(block_size=1024, batch_size=64, reduced=False):
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
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Encoder and Decoder functions
    encode = lambda s: [stoi[c] for c in s]

    # 3. Process data into tensors
    data = torch.tensor(encode(text), dtype=torch.long)

    # 4. Split into Train and Val (90% / 10%)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    ood_data = val_data[:batch_size * block_size]

    def get_batch(split_data):
        # We generate random starting indices for the batch
        ix = torch.randint(len(split_data) - block_size, (batch_size,))
        x = torch.stack([split_data[i:i + block_size] for i in ix])
        y = torch.stack([split_data[i + 1:i + block_size + 1] for i in ix])
        return x, y

    # Note: In a production GPT, you'd return a DataLoader object.
    # Here, we return the processed tensors or a lambda for easy fetching.

    return CharDataset(train_data, block_size), CharDataset(val_data, block_size), CharDataset(ood_data, block_size), vocab_size
    # return train_data, val_data, ood_data, vocab_size


