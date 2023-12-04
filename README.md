# Tracking Anything in Heart All at Once

This repository holds the Pytorch implementation of [Tracking Anything in Heart All at Once](https://cshennju.github.io/NeuralCMF.github.io/). If you find our code useful in your research, please consider citing:

```
@article{shen2023tracking,
  title={Tracking Anything in Heart All at Once},
  author={Shen, Chengkang and Zhu, Hao and Zhou, You and Liu, Yu and Yi, Si and Dong, Lili and Zhao, Weipeng and Brady, David and Cao, Xun and Ma, Zhan and Lin, Yi},
  journal={arXiv preprint arXiv:2310.02792},
  year={2023}
}
```

## Introduction

In this repository, we provide

## Quickstart

This repository is build upon Python v3.8 and Pytorch v1.10.0 on Ubuntu 18.04. All experiments are conducted on a single NVIDIA A100 GPU. See [`requirements.txt`](requirements.txt) for other dependencies. We recommend installing Python v3.8 from [Anaconda](https://www.anaconda.com/) and installing Pytorch (= 1.10.0) following guide on the [official instructions](https://pytorch.org/) according to your specific CUDA version. Then you can install dependencies with the following commands.

```
git clone https://github.com/cshennju/NeuralCMF.git
cd NeuralCMF
pip install -r requirements.txt
```
### STRAUS Datasets

### 3D Echo Datasets

### 2D Echo Datasets


## Acknowledgement
This code is extended from the following repositories.
- [ngp_pl](https://github.com/kwea123/ngp_pl)
- [nsff_pl](https://github.com/kwea123/nsff_pl)

We thank the authors for releasing their code. Please also consider citing their work.
