# Neural Collapse in Quantised Neural Networks

## Description

Modern neural networks not only achieve impressive accuracy but also reveal surprisingly simple geometric structures in their learned representations.
One such phenomenon, called neural collapse, emerges once a network has effectively reached zero training error: 
the network’s internal representations align into elegant, low-dimensional patterns.

This project aims to study the interaction of the neural collapse phenomenon with quantization.

### Goals:
- Experimentally investigate how quantized networks behave in the post–zero-loss regime 
- Utilize geometric and combinatorial tools to understand the structure of both neural collapse and quantization


## Installation

uv


## Usage

### uv

uv run main.py --exp-name test --lr 0.01 --epochs 200 --analysis_freq 20 --save t


Run with ResNet18 on CIFAR10 for 100 epochs
    python train.py --exp_name "test" --model ResNet18 --epochs 100
