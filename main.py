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
from sympy.strategies.core import switch

import plotter
from argparser import get_args
from experiment_logger import ExperimentLogger
from general_analysis import ModelAnalyzer
from quantization.q_controller import apply_quantization_config
from data_loading.dataset_loader import DatasetLoader
from model_architectures.model_constructor import ModelConstructor
from train_utils import set_seed, enable_full_determinism
from training.language_trainer import LanguageTrainer
from training.vision_trainer import VisionTrainer

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


def main():

    args = get_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )

    # Seeding
    set_seed(args.seed)

    enable_full_determinism(args.seed, args.deterministic)

    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cuda.matmul.fp32_precision = 'ieee'

    torch.set_default_dtype(torch.float32)
    torch.set_autocast_dtype(device_type=str(device), dtype=torch.float32)

    # ptdtype = {
    #     'float32': torch.float32,
    #     'bfloat16': torch.bfloat16,
    #     'float16': torch.float16}
    # [dtype]
    # ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    PRECISION_MAP = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }

    # Default values
    config = {
        "exp_name": "NC_Experiment",
        "task": "cv",
        "num_classes": 10,
        "train_val_test_split": [0.8, 0.1, 0.1],
        "batch_size": 128,
        "epochs": 50,
        "lr": 0.001,
        "scheduler": None,
        "weight_decay": 0, #5e-4,
        "model": "simple_cnn",
        "pretrained": True, # Only for downloaded models
        "dataset": "CIFAR10",
        "nc_freq": 5, # NC analysis every X epochs
        "criterion": str(nn.CrossEntropyLoss()),
        "device": device,
        "now": datetime.now().strftime("%m.%d.%Y_%H.%M.%S"),
        "save": False
    }

    if vars(args)["task"] == "nlp":
        config.update(
            {
                "vocab_size": 64,
                "num_classes": 64,
                "n_embd": 168,
                "n_layer": 16,
                "n_head": 8,
                "batch_size": 32,
                "seq_length": 128,
                "block_size": 128,
                "epochs": 5,
                "nc_freq": 1,
                "max_samples_for_nc": 200,  # Limit samples for NC analysis
                "min_samples_per_class": 5,  # Minimum samples per token
                "grad_accumulation_steps": 1,
                "clip_grad": True,
                "max_grad_norm": 1.0,

                # Ignore early positions where the model has too little context inside the window.
                # This removes an intrinsic accuracy ceiling (often ~97–99%) caused by context collisions
                # at the very beginning of each randomly-windowed sequence.
                "loss_ignore_first_n": 64,
                "nc_ignore_first_n": 64,
            }
        )

    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    config.update(args_dict)

    # config["dtype"] = PRECISION_MAP.get(config["precision"], torch.float32)



    logger = ExperimentLogger(config)

    # Data Loading
    dataset_loader = DatasetLoader()
    train_loader, validation_loader, ood_loader,  config['num_classes'] = dataset_loader.get_data(config)

    if config.get("task") == "nlp": config["vocab_size"] = int(config["num_classes"])

    # Construct Model
    model, tokenizer = ModelConstructor.get_model(config)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model parameters:", count_parameters(model))

    # model = apply_quantization_config(model, config)

    if isinstance(model, tuple):
        model, tokenizer = model

    match config.get("criterion", "cross_entropy").lower():
        case 'cross_entropy':
            criterion = nn.CrossEntropyLoss()

    match config.get("optimizer", "adamw").lower():

        case 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=config['lr'],
                momentum=0.9,
                weight_decay=0.0,
            )
        case 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=config["lr"],
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=False,
                weight_decay=0.0,
                decoupled_weight_decay=True,
            )
        case 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config["lr"],
                weight_decay=0.0,
                betas=(0.9, 0.95)
            )
    config["optimizer"] = str(optimizer)

    # def _init_weights(module):
    #     if isinstance(module, nn.Linear):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #         if module.bias is not None:
    #             torch.nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.Embedding):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #
    # model.apply(_init_weights)

    match config.get("scheduler_type").lower():

        case 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, eta_min=config["lr"] / 1000, T_max=config['epochs']
            )
        case 'step':
            if config.get("task") == "nlp":
                e = int(config["epochs"])
                milestones = sorted(
                    {max(1, int(0.60 * e)), max(1, int(0.80 * e)), max(1, int(0.90 * e))}
                )
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=milestones, gamma=0.1
                )
            else:
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    [
                        100,
                        150,
                    ],
                    0.3
                )
        case _:
            scheduler = None
    config["scheduler"] = scheduler

    # Training
    if config["task"] == "cv":
        trainer = VisionTrainer(
            model, train_loader, validation_loader, ood_loader,  config, logger
        )
    elif config["task"] == "nlp":
        trainer = LanguageTrainer(
            model, train_loader, validation_loader, ood_loader, config, logger
        )
    else:
        exit()

    print('\nUpdated configuration:')
    print(config)

    trained_model = trainer.train(
        criterion, optimizer, config["scheduler"]
    )

    if config["save"]:
        for key, item in config.items():
            config[key] = str(item)
        logger.update_config(config)

    # analyzer = ModelAnalyzer(trained_model, config, logger)
    # results = analyzer.run_all(train_loader, validation_loader, ood_loader, criterion)
    # print("\nFinal Analysis Results:")
    # for metric, value in results.items():
    #     print(f"{metric}: {value:.4f}")

    # print(trained_model)
    # print(model.modules())


if __name__ == '__main__':
    main()
