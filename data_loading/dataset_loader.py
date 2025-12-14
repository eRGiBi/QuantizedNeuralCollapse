import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer

from data_loading.shakespeare_char_prepare import get_shakespeare_char_dataloaders
from data_loading.wikitext import prepare_wikitext_dataset


class DatasetLoader:
    """"""

    @staticmethod
    def get_data(config: dict, data_root='./data'):
        """Load a chosen dataset."""
        num_classes = 0

        match config["dataset"].upper():
        
            case 'CIFAR10':

                # transform_train = transforms.Compose([
                #     transforms.RandomCrop(32, padding=4),
                #     transforms.RandomHorizontalFlip(),
                #     transforms.ToTensor(),
                #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                # ])
                # transform_test = transforms.Compose([
                #     transforms.ToTensor(),
                #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                # ])

                transform_train = transforms.Compose([
                    transforms.Resize(64),
                    # transforms.RandomCrop(224, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # 2. ImageNet normalization values
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225)),
                ])

                # For the test/analysis loader:
                transform_test = transforms.Compose([
                    transforms.Resize(64),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225)),
                ])
                
                trainset = torchvision.datasets.CIFAR10(
                    root=data_root,
                    train=True,
                    download=True,
                    transform=transform_train
                )
                analysis_set = torchvision.datasets.CIFAR10(
                    root=data_root,
                    train=False,
                    download=True,
                    transform=transform_test
                )
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
                trainset, analysis_set, num_classes = prepare_wikitext_dataset(
                    tokenizer=GPT2Tokenizer.from_pretrained("gpt2"), batch_size=config["batch_size"]
                )

            case "SHAKESPEARE_CHAR":
                trainset, analysis_set, num_classes = get_shakespeare_char_dataloaders(
                    data_dir=data_root,
                    block_size=config.get("block_size", 1024),
                    batch_size=config["batch_size"]
                )

            case _:
                raise ValueError(f"Dataset is not supported.")
            
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=config["batch_size"], shuffle=True, prefetch_factor=2, num_workers=2
        )

        analysis_loader = torch.utils.data.DataLoader(
            analysis_set, batch_size=config["batch_size"], shuffle=False, prefetch_factor=2, num_workers=2
        )

        return trainloader, analysis_loader, num_classes
