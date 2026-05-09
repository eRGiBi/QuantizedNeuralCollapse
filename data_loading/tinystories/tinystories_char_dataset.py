from collections import defaultdict
from datasets import load_dataset
from torch.utils.data import Dataset, Subset
import torch



class TinyStoriesCharDataset(Dataset):
    """Character-level sliding-window dataset over TinyStories text."""

    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]

class _PoolDataset(Dataset):
    """
    Thin wrapper that exposes only a subset of `base.data` indices as a
    contiguous TinyStoriesCharDataset, so balanced_subsample_chunked can
    scan it without materialising a huge index array.
    """

    def __init__(
        self,
        base: TinyStoriesCharDataset,
        pool_indices: list[int],
        block_size: int,
    ):
        # Map pool windows → original data positions.
        # A pool window i needs data[pool_indices[i] : pool_indices[i]+block_size+1],
        # but pool_indices are dataset (window) indices, not data indices.
        # We extract the contiguous data slice that covers the pool.
        min_idx = min(pool_indices)
        max_idx = max(pool_indices)
        # data slice covers all windows in the pool
        self.data = base.data[min_idx : max_idx + block_size + 1]
        self.block_size = block_size
        # Remap window indices to be relative to min_idx
        self._offsets = torch.tensor(
            [i - min_idx for i in pool_indices], dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        offset = int(self._offsets[idx])
        chunk = self.data[offset : offset + self.block_size + 1]
        return chunk[:-1], chunk[1:]