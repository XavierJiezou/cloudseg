# CDnet: CNN-Based Cloud Detection for Remote Sensing Imagery

> [CDnet: CNN-Based Cloud Detection for Remote Sensing Imagery](https://ieeexplore.ieee.org/document/8681238)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://ieeexplore.ieee.org/document/8681238">Official Repo</a>

<a href="https://github.com/nkszjx/CDnetV2-pytorch-master/tree/main">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

Cloud detection is one of the important tasks for remote sensing image (RSI) preprocessing. In this paper, we utilize the thumbnail (i.e., preview image) of RSI, which contains the information of original multispectral or panchromatic imagery, to extract cloud mask efficiently. Compared with detection cloud mask from original RSI, it is more challenging to detect cloud mask using thumbnails due to the loss of resolution and spectrum information. To tackle this problem, we propose a cloud detection neural network (CDnet) with an encoder–decoder structure, a feature pyramid module (FPM), and a boundary refinement (BR) block. The FPM extracts the multiscale contextual information without the loss of resolution and coverage; the BR block refines object boundaries; and the encoder–decoder structure gradually recovers segmentation results with the same size as input image. Experimental results on the ZY-3 satellite thumbnails cloud cover validation data set and two other validation data sets (GF-1 WFV Cloud and Cloud Shadow Cover Validation Data and Landsat-8 Cloud Cover Assessment Validation Data) demonstrate that the proposed method achieves accurate detection accuracy and outperforms several state-of-the-art methods.


<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/user-attachments/assets/502456ca-0f61-4616-99e7-baa93ee6d5e2" width="70%"/>
</div>

## Results and models

### CLOUD EXTRACTION ACCURACY (%)


| Method  | OA    | MIoU  | Kappa | PA    | UA    |
|----------|-------|-------|-------|-------|-------|
| CDnet(ASPP+GAP) | 95.41 | 89.38 | 82.05 | 87.82 | 89.85 |
| CDnet(FPM) | 96.47 | 91.70 | 85.06 | 89.75 | 90.41 |


### CLOUD EXTRACTION ACCURACY (%) FOR MODULES AND VARIANTS OF THE CDNET



| Method  | OA    | MIoU  | Kappa | PA    | UA    |
|----------|-------|-------|-------|-------|-------|
| ResNet50 | 91.13 | 82.83 | 73.38 | 81.99 | 80.34 |
| MRN*     | 93.03 | 85.24 | 77.51 | 82.59 | 82.82 |
| MRN+FPM  | 93.89 | 88.50 | 81.82 | 87.10 | 85.51 |
| MRN+FPM+BR| 94.31 | 88.97 | 82.59 | 87.12 | 87.04 |
| CDnet-FPM | 93.14 | 88.14 | 80.44 | 87.64 | 84.46 |
| CDnet-BR  | 95.04 | 89.63 | 83.78 | 87.36 | 88.67 |
| CDnet-FPM-BR| 93.10 | 87.91 | 80.01 | 87.01 | 83.84 |
| CDnet-A  | 94.84 | 89.41 | 82.91 | 87.32 | 88.07 |
| CDnet-B  | 95.27 | 90.51 | 84.01 | 88.97 | 89.71 |
| CDnet-C  | 96.09 | 90.73 | 84.27 | 88.74 | 90.28 |
| CDnet     | 96.47 | 91.70 | 85.06 | 89.75 | 90.41 |

MRN stands for the modified ResNet-50.



### CLOUD EXTRACTION ACCURACY (%)


| Method  | OA    | MIoU  | Kappa | PA    | UA    |
|----------|-------|-------|-------|-------|-------|
| Maxlike  | 77.73 | 66.16 | 53.55 | 91.30 | 54.98 |
| SVM      | 78.21 | 66.79 | 54.87 | 91.77 | 56.37 |
| L-unet   | 86.51 | 73.67 | 63.79 | 83.15 | 64.79 |
| FCN-8    | 90.53 | 81.08 | 68.08 | 82.91 | 78.87 |
| MVGG-16  | 92.73 | 86.65 | 78.94 | 88.12 | 81.84 |
| DPN      | 93.11 | 86.73 | 79.05 | 87.68 | 83.96 |
| DeeplabV2 | 93.36 | 87.56 | 79.12 | 87.50 | 84.65 |
| PSPnet   | 94.24 | 88.37 | 81.41 | 86.67 | 89.17 |
| DeeplabV3 | 95.03 | 88.74 | 81.53 | 87.63 | 89.72 |
| DeeplabV3+| 96.01 | 90.45 | 83.92 | 88.47 | 90.03 |
| CDnet    | 96.47 | 91.70 | 85.06 | 89.75 | 90.41 |


### Cloud Extraction Accuracy (%) of GF-1 Satellite Imagery

| Method | OA   | MIoU | Kappa | PA    | UA    |
|----------|-------|-------|-------|-------|-------|
| MFC      | 92.36 | 80.32 | 74.64 | 83.58 | 75.32 |
| L-unet   | 92.44 | 82.39 | 76.26 | 87.61 | 74.98 |
| FCN-8    | 92.61 | 82.71 | 76.45 | 87.45 | 75.61 |
| MVGG-16  | 93.07 | 86.17 | 77.13 | 87.68 | 79.50 |
| DPN      | 93.19 | 86.32 | 77.25 | 86.85 | 80.93 |
| DeeplabV2 | 95.07 | 87.00 | 80.07 | 86.60 | 82.18 |
| PSPnet   | 95.30 | 87.45 | 80.74 | 85.87 | 83.27 |
| DeeplabV3 | 95.95 | 88.13 | 81.05 | 86.36 | 88.72 |
| DeeplabV3+| 96.18 | 89.11 | 82.31 | 87.37 | 89.05 |
| CDnet    | 96.73 | 89.83 | 83.23 | 87.94 | 89.60 |

### Cloud Extraction Accuracy (%) of Landsat-8 Satellite Imagery

| Method | OA   | MIoU | Kappa | PA    | UA    |
|----------|-------|-------|-------|-------|-------|
| Fmask    | 85.21 | 71.52 | 63.01 | 86.24 | 70.38 |
| L-unet   | 90.56 | 77.95 | 68.79 | 79.32 | 78.94 |
| FCN-8    | 90.88 | 78.84 | 71.32 | 76.28 | 82.31 |
| MVGG-16  | 93.31 | 81.59 | 77.08 | 77.29 | 83.00 |
| DPN      | 93.40 | 86.34 | 81.52 | 84.61 | 89.93 |
| DeeplabV2 | 94.11 | 86.90 | 81.63 | 84.93 | 89.87 |
| PSPnet   | 95.43 | 88.29 | 83.12 | 86.98 | 90.59 |
| DeeplabV3 | 96.38 | 90.32 | 84.31 | 89.52 | 91.92 |
| CDnet    | 97.16 | 90.84 | 84.91 | 90.15 | 92.08 |

## Citation

```bibtex
@ARTICLE{8681238,
author={J. {Yang} and J. {Guo} and H. {Yue} and Z. {Liu} and H. {Hu} and K. {Li}},
journal={IEEE Transactions on Geoscience and Remote Sensing},
title={CDnet: CNN-Based Cloud Detection for Remote Sensing Imagery},
year={2019},
volume={57},
number={8},
pages={6195-6211}, doi={10.1109/TGRS.2019.2904868} }
```
