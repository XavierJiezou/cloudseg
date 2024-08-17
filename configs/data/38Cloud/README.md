# 38-Cloud: Cloud Segmentation in Satellite Images

> [38-Cloud: Cloud Segmentation in Satellite Images](https://www.kaggle.com/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images/data)

## Introduction

- [Official Site](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset)
- [Paper Download](https://ieeexplore.ieee.org/document/8547095)
- Data Download: [Kaggle](https://www.kaggle.com/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images/data)

## Abstract

This paper presents a deep-learning based framework for addressing the problem of accurate cloud detection in remote sensing images. This framework benefits from a Fully Convolutional Neural Network (FCN), which is capable of pixel-level labeling of cloud regions in a Landsat 8 image. Also, a gradient-based identification approach is proposed to identify and exclude regions of snow/ice in the ground truths of the training set. We show that using the hybrid of the two methods (threshold-based and deep-learning) improves the performance of the cloud identification process without the need to manually correct automatically generated ground truths. In average the Jaccard index and recall measure are improved by 4.36% and 3.62%, respectively.

## Dataset

The entire images of these scenes are cropped into multiple 384*384 patches to be proper for deep learning-based semantic segmentation algorithms. There are 8400 patches for training and 9201 patches for testing. Each patch has 4 corresponding spectral channels which are Red (band 4), Green (band 3), Blue (band 2), and Near Infrared (band 5). Unlike other computer vision images, these channels are not combined. Instead, they are in their corresponding directories.

![sample](https://github.com/user-attachments/assets/dd8d9b6c-b8ac-4b3f-bf8b-9c9954e9291f)

```
name: 38-Cloud: Cloud Segmentation in Satellite Images
source:  Landsat 8
band: 4
pixel: 384x84
disk: (13GB)
annotation: 
    - 0: Clear
    - 1: cloud
scene: -
```

## Citation

```
@INPROCEEDINGS{38-cloud-1,
  author={S. {Mohajerani} and P. {Saeedi}},
  booktitle={IGARSS 2019 - 2019 IEEE International Geoscience and Remote Sensing Symposium},
  title={{Cloud-Net: An End-To-End Cloud Detection Algorithm for Landsat 8 Imagery}},
  year={2019},
  volume={},
  number={},
  pages={1029-1032},
  doi={10.1109/IGARSS.2019.8898776}
}

@INPROCEEDINGS{38-cloud-2, 
author={S. Mohajerani and T. A. Krammer and P. Saeedi}, 
booktitle={2018 IEEE 20th International Workshop on Multimedia Signal Processing (MMSP)}, 
title={{A Cloud Detection Algorithm for Remote Sensing Images Using Fully Convolutional Neural Networks}}, 
year={2018},  
pages={1-5}, 
doi={10.1109/MMSP.2018.8547095}, 
ISSN={2473-3628}, 
month={Aug},
}
```
