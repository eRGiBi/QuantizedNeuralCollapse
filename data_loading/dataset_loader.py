import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import Normalize, ToTensor, Compose
from transformers import GPT2Tokenizer
from torch.utils.data import random_split

from data_loading.shake_txt_loader import prepare_text_char_dataset
from data_loading.shakespeare_char_prepare import prepare_shakespeare_char_dataset
from data_loading.wikitext import prepare_wikitext_dataset


class DatasetLoader:
    """Helper class to load datasets based on specified names and configurations."""

    @staticmethod
    def get_data(config: dict, data_root='./data'):
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

                trainset = torchvision.datasets.CIFAR100(
                    root=data_root,
                    train=True,
                    download=True,
                    transform=transform
                )
                analysis_set = torchvision.datasets.CIFAR100(
                    root=data_root,
                    train=False,
                    download=True,
                    transform=transform
                )
                num_classes = 100

            case "WIKITEXT":
                train_set, analysis_set, num_classes = prepare_wikitext_dataset(
                    tokenizer=GPT2Tokenizer.from_pretrained("gpt2"),
                    batch_size=config["batch_size"]
                )

            case "kar_SHAKESPEARE_CHAR":
                train_set, val_set, ood_set, num_classes = prepare_shakespeare_char_dataset(
                    block_size=config.get("block_size", 1024),
                    batch_size=config["batch_size"]
                )

            case "SHAKESPEARE_CHAR":
                # train_set, val_set, ood_set, num_classes = prepare_shakespeare_char_dataset(
                train_set, val_set, ood_set, num_classes = prepare_text_char_dataset(
                    block_size=config.get("block_size", 1024),
                    batch_size=config["batch_size"],
                    reduced=True
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
