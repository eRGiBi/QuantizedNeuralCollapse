from collections import Counter
import json
import hashlib
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from torch.utils.data import Dataset, Subset
import torch

from data_loading.tinystories.tinystories_char_dataset import TinyStoriesCharDataset, _PoolDataset


def _build_vocab(text: str) -> dict[str, int]:
    return {ch: i for i, ch in enumerate(sorted(set(text)))}


def _build_filtered_vocab(
    text: str, min_class_count: int
):
    """Build a vocabulary that only contains characters appearing at least
    `min_class_count` times.  Rare characters are dropped from the vocab
    entirely so encoded indices are always in [0, num_classes).

    Returns:
        stoi          – char → contiguous index mapping (no gaps)
        dropped_count – number of characters that were filtered out
    """
    freq = Counter(text)
    valid_chars = sorted(ch for ch, cnt in freq.items() if cnt >= min_class_count)
    dropped = len(set(text)) - len(valid_chars)

    if dropped:
        print(
            f"  Dropped {dropped} character(s) with fewer than "
            f"{min_class_count:,} occurrences — they are excluded from "
            f"the vocabulary and skipped during encoding."
        )

    stoi = {ch: i for i, ch in enumerate(valid_chars)}
    return stoi, dropped


def _encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    # Characters absent from stoi are silently skipped — this is intentional
    # and ensures all indices are in [0, len(stoi)).
    return torch.tensor([stoi[c] for c in text if c in stoi], dtype=torch.long)


def _cache_path(
    cache_dir: Path,
    split: str,
    block_size: int,
    max_per_class: int,
    min_class_count: int,
    seed: int,
) -> Path:
    key = hashlib.md5(
        f"{split}|{block_size}|{max_per_class}|{min_class_count}|{seed}".encode()
    ).hexdigest()[:10]
    return cache_dir / f"tinystories_{split}_{key}.pt"


def _trim_and_report(
        buckets: dict[int, list[int]],
        num_classes: int,
        split_name: str,
) -> list[int]:
    """Trim every bucket to the minimum count across all classes (strict balance)
    and return a flat list of selected window indices.
    """
    quota = min(len(v) for v in buckets.values())
    print(
        f"  [{split_name}] quota={quota:,}/class | "
        f"{quota * num_classes:,} total | "
        f"{num_classes} classes"
    )
    selected: list[int] = []
    for idxs in buckets.values():
        selected.extend(idxs[:quota])
    return selected


def _get_valid_classes(data: torch.Tensor, num_classes: int, min_count: int) -> set[int]:
    """Return the set of class IDs (character indices) whose total frequency in
    `data` is at least `min_count`.  Uses a single vectorised bincount so it
    never builds a Python-side list of the full tensor.
    """
    counts = torch.bincount(data, minlength=num_classes)
    valid = torch.where(counts >= min_count)[0].tolist()
    dropped = num_classes - len(valid)
    if dropped:
        print(
            f"  Dropped {dropped} class(es) with fewer than "
            f"{min_count:,} occurrences in the corpus."
        )
    return set(valid)

def _subsample_train(
    dataset: TinyStoriesCharDataset,
    num_classes: int,
    max_per_class: int,
    rng: torch.Generator,
    chunk_size: int,
) -> list[int]:
    """Scan the training data in randomly-ordered chunks, collecting up to
    `max_per_class` window indices per class.  Stops as soon as every class
    bucket is full (early exit keeps RAM flat even on the 470 M-window set).
    """
    N = len(dataset)
    block_size = dataset.block_size
    num_chunks = (N + chunk_size - 1) // chunk_size

    chunk_order = torch.randperm(num_chunks, generator=rng).tolist()
    buckets = {cls: [] for cls in range(num_classes)}
    classes_done = 0

    for chunk_idx in chunk_order:
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, N)

        # Last predicted token for every window in this chunk
        chunk_labels = dataset.data[start + block_size : end + block_size]

        for cls in range(num_classes):
            if len(buckets[cls]) >= max_per_class:
                continue

            hits = torch.where(chunk_labels == cls)[0]
            if hits.numel() == 0:
                continue

            perm = torch.randperm(hits.numel(), generator=rng)
            needed = max_per_class - len(buckets[cls])
            prev = len(buckets[cls])
            buckets[cls].extend((hits[perm[:needed]] + start).tolist())

            if prev < max_per_class <= len(buckets[cls]):
                classes_done += 1

        if classes_done == num_classes:
            break

    # return _finalise_buckets(buckets, max_per_class, "train")
    return _trim_and_report(buckets, num_classes, "train")


def _finalise_buckets(
    buckets: dict[int, list[int]], max_per_class: int, split_name: str
) -> list[int]:
    if not buckets:
        print(f"  [{split_name}] WARNING: empty buckets — returning []")
        return []

    quota = min(min(len(v) for v in buckets.values()), max_per_class)
    print(
        f"  [{split_name}] quota={quota:,}/class | "
        f"{quota * len(buckets):,} total samples | "
        f"{len(buckets)} classes"
    )
    selected: list[int] = []
    for idxs in buckets.values():
        selected.extend(idxs[:quota])
    return selected


def _subsample_pool(
        data: torch.Tensor,
        block_size: int,
        pool_indices: list[int],
        num_classes: int,
        max_per_class: int,
        rng: torch.Generator,
        split_name: str,
) -> list[int]:
    """Vectorised scan over a pre-defined pool of window indices (val / ood).

    The pool is small enough to fit in memory, so we shuffle it once, look up
    all labels in a single gather, then slice per class — no Python loop over
    individual windows.
    """
    pool = torch.tensor(pool_indices, dtype=torch.long)
    pool = pool[torch.randperm(len(pool), generator=rng)]  # shuffle
    labels = data[pool + block_size]

    buckets: dict[int, list[int]] = {}
    for cls in range(num_classes):
        hits = pool[labels == cls]
        buckets[cls] = hits[:max_per_class].tolist()

    return _trim_and_report(buckets, num_classes, split_name)


def prepare_tinystories_dataset(
    seed,
    rng: torch.Generator,
    block_size: int = 256,
    max_per_class: int = 1_000,
    min_class_count: int = 150,
    val_fraction: float = 0.05,
    ood_fraction: float = 0.05,
    cache_dir: str | Path = "data/tinystories/sub",
    chunk_size: int = 500_000,
    use_cache: bool = True,
) -> tuple[Dataset, Dataset, Dataset, int]:
    """Prepare a class-balanced TinyStories character-level dataset.

    Every split (train / val / ood) has *exactly* the same number of samples
    per class — a hard requirement for neural collapse analysis.  The quota is
    ``min(max_per_class, rarest_class_count)`` and is enforced after scanning
    by trimming every bucket to the minimum, so the balance is always exact.

    Characters with fewer than `min_class_count` total occurrences in the
    training corpus are excluded from the vocabulary entirely; they will never
    appear as a class label.

    Args:
        block_size:      Context length (tokens per sample).
        max_per_class:   Hard cap on samples per class per split.
        min_class_count: Minimum corpus frequency for a character to be
                         treated as a class.  Raise this if the quota is
                         being dragged down by rare symbols.
        val_fraction:    Fraction of HF validation split used for val.
        ood_fraction:    Fraction of HF validation split used for OOD.
        seed:            Master RNG seed (also used as cache key).
        rng:             Optional pre-seeded Generator (overrides `seed`).
        cache_dir:       Directory for cached index files.
        chunk_size:      Labels processed per chunk (tune for your RAM).
        use_cache:       If True, load cached indices when available and
                         skip re-scanning.  Set False to force rebuild.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Raw text + vocab
    raw = load_dataset("roneneldan/TinyStories", trust_remote_code=True)
    train_text = "\n".join(raw["train"]["text"])
    val_text = "\n".join(raw["validation"]["text"])

    # vocab_path = cache_dir / "vocab.json"
    # if not vocab_path.exists():
    #     vocab_path.write_text(
    #         json.dumps(stoi, ensure_ascii=False, indent=2), encoding="utf-8"
    #     )

    print(
        f"[TinyStories] Building filtered vocab "
        f"(min_class_count={min_class_count:,})..."
    )
    stoi, n_dropped = _build_filtered_vocab(train_text, min_class_count)
    num_classes = len(stoi)
    print(f"  {num_classes} characters retained, {n_dropped} dropped.")


    # Encode — rare chars are skipped at this step, not at lookup time
    train_data = _encode(train_text, stoi)
    val_data = _encode(val_text, stoi)

    full_train = TinyStoriesCharDataset(train_data, block_size)
    full_val = TinyStoriesCharDataset(val_data, block_size)

    # Partition the val dataset into val and OOD pools

    # 4. Val / OOD pool indices
    n_val = len(full_val)
    pool_rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_val, generator=pool_rng).tolist()

    val_end = max(1, int(n_val * val_fraction))
    ood_end = val_end + max(1, int(n_val * ood_fraction))

    val_pool_indices = perm[:val_end]
    ood_pool_indices = perm[val_end:ood_end]

    # Subsampling (with caching results)
    def _load_or_build(split: str, build_fn) -> list[int]:
        cp = _cache_path(
            cache_dir, split, block_size, max_per_class, min_class_count, seed
        )
        if use_cache and cp.exists():
            print(f"[TinyStories] Loading cached {split} indices ← {cp}")
            return torch.load(cp, weights_only=True).tolist()
        print(f"[TinyStories] Building balanced {split} split...")
        indices = build_fn()

        torch.save(torch.tensor(indices, dtype=torch.int64), cp)
        print(f"[TinyStories] Cached {split} indices → {cp}")

        return indices

    train_indices = _load_or_build(
        "train",
        lambda: _subsample_train(
            full_train,
            num_classes=num_classes,
            max_per_class=max_per_class,
            rng=torch.Generator().manual_seed(seed),
            chunk_size=chunk_size,
        ),
    )
    val_indices = _load_or_build(
        "val",
        lambda: _subsample_pool(
            val_data,
            block_size=block_size,
            pool_indices=val_pool_indices,
            num_classes=num_classes,
            max_per_class=max_per_class,
            rng=torch.Generator().manual_seed(seed + 1),
            split_name="val",
        ),
    )
    ood_indices = _load_or_build(
        "ood",
        lambda: _subsample_pool(
            val_data,
            block_size=block_size,
            pool_indices=ood_pool_indices,
            num_classes=num_classes,
            max_per_class=max_per_class,
            rng=torch.Generator().manual_seed(seed + 2),
            split_name="ood",
        ),
    )

    return (
        Subset(full_train, train_indices),
        Subset(full_val, val_indices),
        Subset(full_val, ood_indices),
        num_classes,
    )

def balanced_subsample(
    dataset: TinyStoriesCharDataset,
    max_per_class: int,
    rng,
) -> Subset:
    """Return a Subset in which every class (last target token) has exactly
    `min(max_per_class, rarest_class_count)` samples.

    Args:
        dataset:       A TinyStoriesCharDataset (must expose `.data`).
        max_per_class: Hard cap on samples per class. The actual cap is
                       min(max_per_class, rarest_class_count), so balance
                       is always exact.


    Returns:
        A class-balanced Subset.
    """
    # Bucket every index by its class label
    # data[idx + block_size] is the last token the window must predict
    # Vectorised: labels[i] == dataset.class_label(i) for all i
    labels: torch.Tensor = dataset.data[dataset.block_size :]  # shape (N,)

    buckets = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        buckets[label].append(idx)

    # Determine the per-class quota                                     #
    rarest = min(len(v) for v in buckets.values())
    quota = min(max_per_class, rarest)

    if quota < rarest:
        print(
            f"[balanced_subsample] Capping at {quota:,} samples/class "
            f"(rarest class has {rarest:,}). "
            f"Total samples: {quota * len(buckets):,} across {len(buckets)} classes."
        )
    else:
        print(
            f"[balanced_subsample] Rarest class ({rarest:,}) is the limiting factor. "
            f"Total samples: {quota * len(buckets):,} across {len(buckets)} classes."
        )

    # Randomly draw `quota` indices from each bucket
    selected: list[int] = []

    for indices in buckets.values():
        t = torch.tensor(indices)
        perm = torch.randperm(len(t), generator=rng)
        selected.extend(t[perm[:quota]].tolist())

    return Subset(dataset, selected)


def _balanced_subsample_chunked(
    dataset: TinyStoriesCharDataset,
    num_classes: int,
    max_per_class: int,
    rng: torch.Generator,
    chunk_size: int = 500_000,
) -> list[int]:
    """
    Scan the dataset in shuffled chunks and collect up to `max_per_class`
    indices per class (defined as the last target token of each window).

    Memory cost is O(chunk_size) instead of O(N), and scanning stops as
    soon as every bucket is full — so in practice far fewer than N windows
    are ever examined.
    """
    N = len(dataset)
    block_size = dataset.block_size

    # Shuffle chunk order so we draw uniformly rather than always from the
    # start of the file (important: randperm over #chunks, not over N).
    num_chunks = (N + chunk_size - 1) // chunk_size
    chunk_order = torch.randperm(num_chunks, generator=rng).tolist()

    buckets: dict[int, list[int]] = defaultdict(list)
    classes_done = 0  # how many classes have reached their quota

    for chunk_idx in chunk_order:
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, N)

        # Labels for window i = data[i + block_size]  (last token predicted)
        chunk_labels = dataset.data[start + block_size : end + block_size]

        for cls in range(num_classes):
            if len(buckets[cls]) >= max_per_class:
                continue

            local_indices = torch.where(chunk_labels == cls)[0]
            if local_indices.numel() == 0:
                continue

            # Shuffle within the chunk so repeated calls stay random
            perm = torch.randperm(local_indices.numel(), generator=rng)
            global_indices = (local_indices[perm] + start).tolist()

            needed = max_per_class - len(buckets[cls])
            prev_len = len(buckets[cls])
            buckets[cls].extend(global_indices[:needed])

            if prev_len < max_per_class <= len(buckets[cls]):
                classes_done += 1

        if classes_done == num_classes:
            break  # every bucket is full — no need to scan further

    # Report actual quota (may be < max_per_class for rare classes)
    min_count = min(len(v) for v in buckets.values())
    quota = min(max_per_class, min_count)

    print(
        f"  quota={quota:,}/class, "
        f"{quota * num_classes:,} total samples, "
        f"{num_classes} classes"
    )

    selected: list[int] = []
    for indices in buckets.values():
        selected.extend(indices[:quota])

    return selected


def prepare_full_tinystories_dataset(
    rng,
    block_size: int = 256,
    max_per_class: int = 5_000,
    val_fraction: float = 0.05,
    ood_fraction: float = 0.05,

) -> tuple[Dataset, Dataset, Dataset, int]:
    """Prepare a class-balanced TinyStories dataset.

    The training set is subsampled so every character class contributes
    exactly `min(max_per_class, rarest_class_count)` sequences.

    Val and OOD sets are also balanced (with the same quota) so that NC
    metrics are comparable across splits.

    Args:
        block_size:      Context length (tokens per sample).
        max_per_class:   Maximum training/val/ood sequences per class.
        val_fraction:    Fraction of the HF validation split used for val.
        ood_fraction:    Fraction of the HF validation split used for OOD
                         (drawn from a non-overlapping slice).

    Returns:
        train_set, val_set, ood_set, num_classes
    """
    raw = load_dataset(
        "roneneldan/TinyStories",
        trust_remote_code=True,
        cache_dir="data/tinystories/",
    )
    train_text = "\n".join(raw["train"]["text"])
    val_text = "\n".join(raw["validation"]["text"])

    # Build character vocabulary from training corpus only
    chars = sorted(set(train_text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    num_classes = len(chars)

    # Construct full datasets                                           #
    full_train = TinyStoriesCharDataset(train_text, block_size, stoi)
    full_val = TinyStoriesCharDataset(val_text, block_size, stoi)

    # Carve val and OOD from non-overlapping slices of the validation text
    n_val = len(full_val)
    perm = torch.randperm(n_val, generator=rng).tolist()

    val_end = max(1, int(n_val * val_fraction))
    ood_end = val_end + max(1, int(n_val * ood_fraction))

    val_pool = TinyStoriesCharDataset(val_text, block_size, stoi)
    ood_pool = TinyStoriesCharDataset(val_text, block_size, stoi)

    # 4. Class-balanced subsampling on every split
    print("[prepare_tinystories_dataset] Subsampling train split...")
    train_set = balanced_subsample(
        full_train, rng=rng, max_per_class=max_per_class
    )

    print("[prepare_tinystories_dataset] Subsampling val split...")
    val_set = balanced_subsample(
        Subset(val_pool, perm[:val_end]),  # type: ignore[arg-type]
        max_per_class=max_per_class,
        rng=rng
    )

    print("[prepare_tinystories_dataset] Subsampling OOD split...")
    ood_set = balanced_subsample(
        Subset(ood_pool, perm[val_end:ood_end]),  # type: ignore[arg-type]
        max_per_class=max_per_class,
        rng=rng
    )

    return train_set, val_set, ood_set, num_classes

#
#
# def _subsample_pool(
#     data: torch.Tensor,
#     block_size: int,
#     pool_indices: list[int],
#     valid_classes: set[int],
#     max_per_class: int,
#     rng: torch.Generator,
#     split_name: str,
# ) -> list[int]:
#     """For the small val/ood pools, materialise all labels in one vectorised op
#     and bucket directly — no chunking needed.
#
#     `pool_indices` are window indices into the dataset whose backing tensor
#     is `data`; label of window i = data[i + block_size].
#     """
#     idx_t = torch.tensor(pool_indices, dtype=torch.long)
#     labels = data[idx_t + block_size]  # shape (pool_size,)
#
#     buckets: dict[int, list[int]] = {cls: [] for cls in valid_classes}
#     for win_idx, label in zip(pool_indices, labels.tolist()):
#         if label in buckets:
#             buckets[label].append(win_idx)
#
#     # A class might be absent from this small pool — drop it so it doesn't
#     # drag the quota to 0.
#     present = {cls: idxs for cls, idxs in buckets.items() if idxs}
#     absent = len(valid_classes) - len(present)
#     if absent:
#         print(f"  {absent} class(es) absent from {split_name} pool — excluded.")
#
#     # Shuffle each bucket before capping
#     for cls, idxs in present.items():
#         t = torch.tensor(idxs)
#         perm = torch.randperm(len(t), generator=rng)
#         present[cls] = t[perm].tolist()
#
#     return _finalise_buckets(present, max_per_class, split_name)

