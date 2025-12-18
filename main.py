import os
import time
import math
import pickle
from contextlib import nullcontext

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
from training.language_trainer import LanguageTrainer
from training.vision_trainer import VisionTrainer

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


def main():

    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    # Seeding
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cuda.matmul.fp32_precision = 'ieee'

    torch.set_default_dtype(torch.float32)
    torch.set_autocast_dtype(device_type=str(device), dtype=torch.float32)

    # note: float16 data type will automatically use a GradScaler
    # ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    # ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Default values
    config = {
        "exp_name": "NC_Experiment",
        "task": "cv",
        "num_classes": 10,
        "batch_size": 128,
        "epochs": 50,
        "lr": 0.001,
        "scheduler": None,
        "weight_decay": 0, #5e-4,
        "model": "simple_cnn",
        "pretrained": True, # Only for downloaded models
        "dataset": "CIFAR10",
        "nc_freq": 5, # NC analysis every X epochs
        "criterion": str(nn.CrossEntropyLoss()), # if args.criterion == "cross_entropy" else nn.,
        "device": device,
        "now": datetime.now().strftime("%m.%d.%Y_%H.%M.%S"),
        "save": False
    }
    if vars(args)["task"] == "nlp":
        config.update(
            {
                "vocab_size": 65,  # Smaller for testing
                "num_classes": 65,
                "n_embd": 1024,
                "n_layer": 4,
                "n_head": 4,
                "batch_size": 24,
                "seq_length": 64,
                "block_size": 256,
                "epochs": 5,
                "nc_freq": 1,
                "max_samples_for_nc": 1000,  # Limit samples for NC analysis
                "min_samples_per_class": 50,  # Minimum samples per token
                "grad_accumulation_steps": 1,
                "clip_grad": False,
                "max_grad_norm": 1.0,
            }
        )
    config.update(vars(args))

    print('\nUpdated configuration:')
    print(config)

    logger = ExperimentLogger(config)

    # Data Loading
    dataset_loader = DatasetLoader()
    train_loader, validation_loader, ood_loader,  config['num_classes'] = dataset_loader.get_data(config)

    # Model
    model, tokenizer = ModelConstructor.get_model(config)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model parameters:",  count_parameters(model))

    if isinstance(model, tuple):
        model, tokenizer = model

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['lr'],
        momentum=0.9,
        weight_decay=config['weight_decay']
    )
    # optimizer = optim.Adam(
    #     model.parameters(),
    #     lr=config["lr"],
    #     betas=(0.9, 0.999),
    #     eps=1e-08,
    #     amsgrad=False,
    #     weight_decay=config["weight_decay"],
    #     decoupled_weight_decay=True,
    # )
    config["optimizer"] = str(optimizer)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, eta_min=config["lr"] / 100, T_max=config['epochs']
    # )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        [
            30, #config["epochs"] // 3,
            45, #config["epochs"] * 2 // 3
            # 75,
            # 115,
        ],
        0.1
    )
    config["scheduler"] = scheduler

    # Training
    if config["task"] == "cv":
        trainer = VisionTrainer(model, train_loader, validation_loader, ood_loader,  config, logger)
    elif config["task"] == "nlp":
        trainer = LanguageTrainer(model, train_loader, validation_loader, ood_loader, config, logger)
    else:
        exit()

    trained_model = trainer.train(
        criterion, optimizer, config["scheduler"]
    )

    if config["save"]:
        for key, item in config.items():
            config[key] = str(item)
        logger.update_config(config)

    # logger.save_training_image()

    # print(trained_model)
    # print(model.modules())


if __name__ == '__main__':
    main()
