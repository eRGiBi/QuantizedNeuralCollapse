from datetime import datetime
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from argparser import get_args
from experiment_logger import ExperimentLogger
from data_loading.dataset_loader import DatasetLoader
from model_architectures.model_constructor import ModelConstructor
from training.vision_trainer import VisionTrainer


def main():
    
    args = get_args()
    
    # Seeding
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Default values
    config = {
        "exp_name": "NC_Experiment",
        "task": "cv",
        "num_classes": 10,
        "batch_size": 128,
        "epochs": 50,
        "lr": 0.001,
        "scheduler": None,
        "weight_decay": 5e-4,
        "model": "simple_cnn",
        "pretrained": True, # Only for downloaded models
        "dataset": "CIFAR10",
        "nc_freq": 5, # NC analysis every X epochs
        "criterion": str(nn.CrossEntropyLoss()), # if args.criterion == "cross_entropy" else nn.,
        "device": torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"),
        "now": datetime.now().strftime("%m.%d.%Y_%H.%M.%S"),
        "save": False
    }
    config.update(vars(args))
    
    print('Updated configuration:\n')
    print(config)
    
    if config["save"]:
        logger = ExperimentLogger(config)
    else:
        logger = None
        
    # Data Loading
    dataset_loader = DatasetLoader()
    train_loader, analysis_loader, config['num_classes'] = dataset_loader.get_data(config)

    # Model
    model = ModelConstructor.get_model(config)

    if isinstance(model, tuple):
        model, tokenizer = model

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.999),
        eps=1e-08,
        amsgrad=False,
        weight_decay=config["weight_decay"],
        decoupled_weight_decay=False,
    )
    config["optimizer"] = str(optimizer)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=config['epochs']
    # )
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        [
            config["epochs"] // 3,
            config["epochs"] * 2 // 3
        ],
        0.1
    )

    config["scheduler"] = lr_scheduler

    # Training
    if config["task"] == "cv":
        trainer = VisionTrainer(model, train_loader, analysis_loader,  config, logger)
    # elif config["task"] == "nlp":
    #     trainer = LanguageTrainer(model, train_loader, analysis_loader, config, logger)

    trained_model = trainer.train(
        criterion, optimizer, config["scheduler"]
    )

    if config["save"]:
        for key, item in config.items():
            config[key] = str(item)
        logger.update_config(config)

    # print(trained_model)
    # print(model.modules())


if __name__ == '__main__':
    main()
