import time
from datetime import datetime
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from data_loading.dataset_loader import DatasetLoader
from model_architectures.model_constructor import ModelConstructor
from training import ModelTrainer
from neural_collapse_analyzer import NeuralCollapseAnalyzer

from argparser import get_args
from experiment_logger import ExperimentLogger

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
        "learning_rate": 0.01,
        "weight_decay": 0,
        "model": "simple_cnn",
        "dataset": "CIFAR10",
        "analysis_frequency": 5, # NC analysis every X epochs
        "criterion": str(nn.CrossEntropyLoss()), # if args.criterion == "cross_entropy" else nn.KLDivLoss,
        "device": torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"),
        "now": datetime.now().strftime("%m.%d.%Y_%H.%M.%S"),
        "save": False
    }
    
    config.update(vars(args))
    
    print('Updated configuration:')
    print(config)
    
    if config["save"]:
        logger = ExperimentLogger(config)
    else:
        logger = None
        
    # --- 1. Data Loading ---
    dataset_loader = DatasetLoader()
    trainloader, analysis_loader, config['num_classes'] = dataset_loader.get_data(config)

    # Model
    construction = ModelConstructor.get_model(config)
    if isinstance(construction, tuple):
        model, tokenizer = construction

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-08,
        amsgrad=False,
        weight_decay=config["weight_decay"],
        decoupled_weight_decay=False,
    )
    config["optimizer"] = str(optimizer)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    config["scheduling"] = str(scheduler)

    if config["save"]:
        logger.update_config(config)
    
    # Training Loop
    trained_model = ModelTrainer(model, trainloader, analysis_loader,  config, logger).train(
        criterion, optimizer, scheduler
    )
    

if __name__ == '__main__':
    main()
