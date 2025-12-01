import os
import argparse
from distutils.util import strtobool


def get_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser(description="Run Neural Collapse experiments.")
    
    parser.add_argument("--exp-name", type=str, default="NC_Experiment", help="the name of this experiment")
    # parser.add_argument('--run_type', type=str, default='full', choices=["full", "cont", "test", "saved", "learning"]

    parser.add_argument('--seed', '-s', type=int, default=476, help="Seed of the experiment.")

    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"])

    parser.add_argument('--task', type=str, default='cv', choices=['nlp', 'cv'], help='')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'MNIST', 'wikitext'], help='Dataset to use.')
    parser.add_argument('--model', type=str, default='simple_cnn', choices=["simple_cnn", 'ResNet18', 'GPT2'], help='Model architecture to use.')

    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for the optimizer.')
    parser.add_argument('--analysis_freq', type=int, default=5, help='Frequency (in epochs) to run NC analysis.')
    
    # Saving
    parser.add_argument('--save', default=False, type=lambda x: bool(strtobool(x)))
    parser.add_argument('--savemodel', default=False, type=lambda x: bool(strtobool(x)))
    
     # Wandb
    # parser.add_argument("--wandb", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        # help="If toggled, the experiment will be tracked with Weights and Biases")
    # parser.add_argument("--wandb-entity", type=str, default=None, help="The entity (team) of wandb's project")
    # parser.add_argument('--wandb_rootlog', type=str, default="/wandb")
    
    return parser.parse_args()

