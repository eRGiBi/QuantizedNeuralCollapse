from os.path import exists
from typing import Dict

import torch
from torch.utils.data import random_split, DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import Normalize, ToTensor, Compose

from datasets import load_dataset, load_from_disk
# from datasets.arrow_dataset import Dataset
from transformers import GPT2Tokenizer
from transformers import AutoTokenizer, TrainingArguments
from transformers.testing_utils import CaptureLogger
from transformers.utils import logging

from data_loading.tinystories.tinystories_builder import prepare_tinystories_dataset
from data_loading.shake_txt_loader import prepare_text_char_dataset
from data_loading.shakespeare_char_prepare import prepare_shakespeare_char_dataset
from data_loading.wikitext import prepare_wikitext_dataset

class DatasetLoader:
    """Load datasets based on specified names and configurations."""

    @staticmethod
    def get_data(config: dict,  rng: torch.Generator, data_root='./data',):
        """Load a chosen dataset."""
        num_classes = config["num_classes"]

        match config["dataset"].upper():

            case "MNIST":
                num_classes = 10
                transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

                train_set = MNIST("./data", True, transform, download=True)
                val_set = MNIST("./data", False, transform, download=True)
                ood_set = FashionMNIST("./data", False, transform, download=True)

            case 'CIFAR10':

                # transform_train = transforms.Compose([
                #     transforms.ToTensor(),
                #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                # ])
                # transform_test = transforms.Compose([
                #     transforms.ToTensor(),
                #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                # ])

                transform_train = transforms.Compose([
                    transforms.Resize(64),
                    transforms.ToTensor(),
                    # ImageNet normalization values
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225)),
                ])

                transform_test = transforms.Compose([
                    transforms.Resize(64),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225)),
                ])
                
                train_set = torchvision.datasets.CIFAR10(
                    root=data_root,
                    train=True,
                    download=True,
                    transform=transform_train
                )
                val_set = torchvision.datasets.CIFAR10(
                    root=data_root,
                    train=False,
                    download=True,
                    transform=transform_test
                )

                ood_set, val_set = random_split(val_set, [5000, 5000])

                num_classes = 10

            case "CIFAR100":
                transform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225)),
                ])

                train_set = torchvision.datasets.CIFAR100(
                    root=data_root,
                    train=True,
                    download=True,
                    transform=transform
                )
                val_set = torchvision.datasets.CIFAR100(
                    root=data_root,
                    train=False,
                    download=True,
                    transform=transform
                )
                # use a split for an OOD-like holdout; caller can override as needed
                ood_set, val_set = random_split(val_set, [5000, 5000])
                num_classes = 100

            case "WIKITEXT":
                train_loader, analysis_loader, num_classes = prepare_wikitext_dataset(
                    tokenizer=GPT2Tokenizer.from_pretrained("gpt2"),
                    batch_size=config["batch_size"]
                )

                # `prepare_wikitext_dataset` already returns ready-to-use DataLoaders.
                # Return them directly to avoid wrapping a DataLoader inside another DataLoader.
                return train_loader, analysis_loader, analysis_loader, num_classes

            case "KAR_SHAKESPEARE_CHAR":
                train_set, val_set, ood_set, num_classes = prepare_shakespeare_char_dataset(
                    block_size=config.get("block_size", 1024),
                    batch_size=config["batch_size"],
                    train_split_size=int(config.get("train_split_size", 10000)),
                    deterministic_train=bool(config.get("deterministic_train", False)),
                )

            case "SHAKESPEARE_CHAR":
                # train_set, val_set, ood_set, num_classes = prepare_shakespeare_char_dataset(
                train_set, val_set, ood_set, num_classes = prepare_text_char_dataset(
                    config=config,
                    reduced=True
                )

                # Optional: limit number of training sequences for overfitting / terminal-phase NC.
                # (train_split_size is already exposed in argparser, but was previously ignored here.)
                tss = config.get("train_split_size", None)
                if tss is not None:
                    tss = int(tss)
                    if tss > 0 and tss < len(train_set):
                        train_set = Subset(train_set, list(range(tss)))

            case "TINYSTORIES":
                train_set, val_set, ood_set, num_classes = prepare_tinystories_dataset(
                    block_size=config.get("block_size", 256),
                    max_per_class=int(config.get("max_per_class", 5_000)),
                    val_fraction=float(config.get("val_fraction", 0.05)),
                    ood_fraction=float(config.get("ood_fraction", 0.05)),
                    seed=config.get("seed"),
                    rng=rng,
                    use_cache=True
                )
            case _:
                raise ValueError(f"Dataset is not supported.")

        batch_size = config["batch_size"]

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        ood_loader = DataLoader(
            ood_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        return train_loader, val_loader, ood_loader, num_classes


def get_data_as_chunks(
    args, cache_dir: str, token: str
) -> Dict[str, Dataset]:
    """
    Get the datasets: you can either provide your own CSV/JSON/TXT training and
    evaluation files (see below) or just provide the name of one of the public
    datasets available on the hub at https://huggingface.co/datasets/ (the
    dataset will be downloaded automatically from the datasets Hub).

    For CSV/JSON files, this script will use the column called 'text' or the
    first column if no column called 'text' is found. You can easily tweak this
    behavior (see below).

    In distributed training, the load_dataset function guarantee that only one
    local process can concurrently download the dataset.
    """

    if args.data_dir is not None and exists(args.data_dir):
        raw_datasets = load_from_disk(args.data_dir)
    elif args.dataset_name is not None:
        raw_datasets = load_dataset("roneneldan/TinyStories")
    elif args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=cache_dir,
            token=token,
            streaming=args.streaming,
        )

        if args.data_dir is not None:
            raw_datasets.save_to_disk(args.data_dir)

        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=cache_dir,
                token=token,
                streaming=args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=cache_dir,
                token=token,
                streaming=args.streaming,
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (
            args.train_file.split(".")[-1]
            if args.train_file is not None
            else args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=cache_dir,
            token=token,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=cache_dir,
                token=token,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=cache_dir,
                token=token,
                **dataset_args,
            )

    return raw_datasets


def tokenize_dataset(
    train_args: TrainingArguments,
    data_args,
    tokenizer: AutoTokenizer,
    raw_datasets: Dict[str, Dataset],
):
    """Tokenize the entire dataset.
    train_args: Training arguments supplied from top-level script.
    data_args: Dataset arguments supplied from top-level script.
    tokenizer: Tokenizer model to process tokens.
    raw_datasets: unprocessed data.
    """
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if train_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    with train_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )

    return tokenized_datasets
