import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer

from data_loading.wikitext import prepare_wikitext_dataset


class DatasetLoader:

    @staticmethod
    def get_data(config: dict, data_root='./data'):
        """Load a chosen dataset.
        
        Args:
            dataset_name (str): The name of the dataset ('CIFAR10', etc.).
            data_root (str): The root directory for the dataset.

        Returns:
            tuple: (trainloader, analysis_loader, num_classes)
        """
        match config["dataset"].upper():
        
            case 'CIFAR10':
                
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
                
                trainset = torchvision.datasets.CIFAR10(
                    root=data_root,
                    train=True,
                    download=True,
                    transform=transform_train
                )
                analysis_set = torchvision.datasets.CIFAR10(
                    root=data_root,
                    train=True,
                    download=True,
                    transform=transform_test
                )
                num_classes = 10
                
            case "WIKITEXT":
                trainset, analysis_set, num_classes = prepare_wikitext_dataset(
                    tokenizer=GPT2Tokenizer.from_pretrained("gpt2"), batch_size=config["batch_size"]
                )

            case _:
                raise ValueError(f"Dataset is not supported.")
            
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=config["batch_size"], shuffle=True, num_workers=2
        )

        analysis_loader = torch.utils.data.DataLoader(
            analysis_set, batch_size=config["batch_size"], shuffle=False, num_workers=2
        )

        return trainloader, analysis_loader, num_classes


