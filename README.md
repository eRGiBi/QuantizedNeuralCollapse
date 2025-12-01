# Neural Collapse in Quantised Neural Networks

## Description

Modern neural networks not only achieve impressive accuracy but also reveal surprisingly simple geometric structures in their learned representations.
One such phenomenon, called neural collapse, emerges once a network has effectively reached zero training error: 
the network’s internal representations align into elegant, low-dimensional patterns.

This project aims to study the interaction of the neural collapse phenomenon with quantization.

---

### Goals:
- Experimentally investigate how quantized networks behave in the post–zero-loss regime 
- Utilize geometric and combinatorial tools to understand the structure of both neural collapse and quantization

---

## Installation

Install dependencies specified in the included uv lockfile to in virtual env, for example:

    uv init

    uv sync 

---

## Usage

### uv



     uv run main.py --exp-name test --model simple_cnn --lr 0.0075 --epochs 200 --nc_freq 10 --save t

