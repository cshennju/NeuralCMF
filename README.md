# Tracking Anything in Heart All at Once

This repository holds the Pytorch implementation of [Tracking Anything in Heart All at Once](https://njuvision.github.io/NeuralCMF/). If you find our code useful in your research, please consider citing:

```
@article{shen2023tracking,
  title={Tracking Anything in Heart All at Once},
  author={Shen, Chengkang and Zhu, Hao and Zhou, You and Liu, Yu and Yi, Si and Dong, Lili and Zhao, Weipeng and Brady, David and Cao, Xun and Ma, Zhan and Lin, Yi},
  journal={arXiv preprint arXiv:2310.02792},
  year={2023}
}
```

## Introduction

In this repository, detailed examples are provided to demonstrate the application of our code across three unique echocardiogram video datasets: the STRAUS Datasets, and both 2D and 3D Echocardiogram Video Datasets.

## Quickstart

This repository is build upon Python v3.8 and Pytorch v1.10.0 on Ubuntu 18.04. All experiments are conducted on a single NVIDIA A100 GPU. See [`requirements.txt`](requirements.txt) for other dependencies. We recommend installing Python v3.8 from [Anaconda](https://www.anaconda.com/) and installing Pytorch (= 1.10.0) following guide on the [official instructions](https://pytorch.org/) according to your specific CUDA version. Then you can install dependencies with the following commands.

```
git clone https://github.com/cshennju/NeuralCMF.git
cd NeuralCMF
pip install -r requirements.txt
```
### STRAUS Datasets
The open-source 3D Strain Assessment in Ultrasound (STRAUS) dataset [1] consists of 8 distinct volumetric sequences, each corresponding to a specific physiological condition. The 3D video in each data has a resolution (130, 110, 140). All volumetric sequences are organized and readily available in the [straus folder](straus).

```
git clone https://github.com/cshennju/NeuralCMF.git
```

### 3D Echo Datasets

```
git clone https://github.com/cshennju/NeuralCMF.git
```

### 2D Echo Datasets

```
git clone https://github.com/cshennju/NeuralCMF.git
```

## References
[1] Alessandrini, M., De Craene, M., Bernard, O., Giffard-Roisin, S., Allain, P., Waechter-Stehle, I., ... & D'hooge, J. (2015). [A pipeline for the generation of realistic 3D synthetic echocardiographic sequences: Methodology and open-access database.](https://ieeexplore.ieee.org/abstract/document/7024160) IEEE transactions on medical imaging, 34(7), 1436-1451.


## Acknowledgement
This code is extended from the following repositories.
- [ngp_pl](https://github.com/kwea123/ngp_pl)
- [nsff_pl](https://github.com/kwea123/nsff_pl)

We thank the authors for releasing their code. Please also consider citing their work.
