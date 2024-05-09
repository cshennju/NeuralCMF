# Tracking Anything in Heart All at Once

This repository holds the Pytorch implementation of [Continuous 3D Myocardial Motion Tracking via Echocardiography](https://njuvision.github.io/NeuralCMF/). If you find our code useful in your research, please consider citing:

```
@article{shen2023tracking,
  title={Continuous 3D Myocardial Motion Tracking via Echocardiography},
  author={Shen, Chengkang and Zhu, Hao and Zhou, You and Liu, Yu and Yi, Si and Dong, Lili and Zhao, Weipeng and Brady, David and Cao, Xun and Ma, Zhan and Lin, Yi},
  journal={arXiv preprint},
  year={2024}
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
#### :key: Training
```
python train_heart_dy_3d.py --root_dir './straus/normal' --exp_name 'straus/normal' --dataset_name 'heart_dy_3d' --T 4 --img_size [130,110,140]
```

### 3D Echo Datasets
The 3D echocardiogram video data was acquired for each volunteer using commercial PHILIPS EPIQ 7C machines equipped with a X5-1 3DE probe. These 3D images were captured employing a standard four-chamber apical view. The 3D video in each data has a resolution (160, 160, 160). The example data are organized and readily available in the [3d_example folder](3d_example).
#### :key: Training
```
python train_heart_dy_3d.py --root_dir './3d_example/' --exp_name '3d_example' --dataset_name 'heart_dy_3d' --T 4 --img_size [160,160,160]
```

### 2D Echo Datasets
The 2D echocardiogram video data was acquired using commercial PHILIPS EPIQ 7C/IE ELITE machines with S5-1 2DE probes and SIEMENS ACUSON SC2000 PRIME machines with a 4V1c 2DE probe. During the imaging process, sonographers performed 360-degree rotations around the apex of the heart for each individual volunteer. Subsequently, the 2D echocardiogram videos were synchronized based on concurrently recorded ECG signals. The resolution of each image is (160, 160). The example data are organized and readily available in the [2d_example folder](2d_example).
#### :key: Training
```
python train_heart_dy.py --root_dir './2d_example/' --exp_name '2d_example' --dataset_name 'heart_dy' --T 6 --img_size [160,160,160]
```

## References
[1] Alessandrini, M., De Craene, M., Bernard, O., Giffard-Roisin, S., Allain, P., Waechter-Stehle, I., ... & D'hooge, J. (2015). [A pipeline for the generation of realistic 3D synthetic echocardiographic sequences: Methodology and open-access database.](https://ieeexplore.ieee.org/abstract/document/7024160) IEEE transactions on medical imaging, 34(7), 1436-1451.


## Acknowledgement
This code is extended from the following repositories.
- [ngp_pl](https://github.com/kwea123/ngp_pl)
- [nsff_pl](https://github.com/kwea123/nsff_pl)

We thank the authors for releasing their code. Please also consider citing their work.
