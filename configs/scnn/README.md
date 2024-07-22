# Remote sensing image cloud detection using a shallow convolutional neural network

> [Remote sensing image cloud detection using a shallow convolutional neural network](https://www.sciencedirect.com/science/article/abs/pii/S0924271624000352?via%3Dihub#fn1)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/1921134176/Deeplearning-for-cloud-detection">Official Repo</a>

<a href="https://github.com/dfchai/SCNN">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

The state-of-the-art methods for cloud detection are dominated by deep convolutional neural networks (DCNNs). However, it is very expensive to train DCNNs for cloud detection and the trained DCNNs do not always perform well as expected. This paper proposes a shallow CNN (SCNN) by removing pooling/unpooling layers and normalization layers in DCNNs, retaining only three convolutional layers, and equipping them with 3 filters of 
1
×
1
,
1
×
1
,
3
×
3
 in spatial dimensions. It demonstrates that the three convolutional layers are sufficient for cloud detection. Since the label output by the SCNN for a pixel depends on a 3 × 3 patch around this pixel, the SCNN can be trained using some thousands 3 × 3 patches together with ground truth of their center pixels. It is very cheap to train a SCNN using some thousands 3 × 3 patches and to provide ground truth of their center pixels. Despite of its low cost, SCNN training is stabler than DCNN training, and the trained SCNN outperforms the representative state-of-the-art DCNNs for cloud detection. The same resolution of original image, feature maps and final label map assures that details are not lost as by pooling/unpooling in DCNNs. The border artifacts suffering from deep convolutional and pooling/unpooling layers are minimized by 3 convolutional layers with 
1
×
1
,
1
×
1
,
3
×
3
 filters. Incoherent patches suffering from patch-by-patch segmentation and batch normalization are eliminated by SCNN without normalization layers. Extensive experiments based on the L7 Irish, L8 Biome and GF1 WHU datasets are carried out to evaluate the proposed method and compare with state-of-the-art methods. The proposed SCNN promises to deal with images from any other sensors.


## Results and models

### 

| Dataset  | Zone/Biome | Clear |       |       |       | Cloud |       |       |
|----------|------------|-------|-------|-------|-------|-------|-------|-------|
|          |            | A^O   | A^P   | A^U   | F_1   | A^P   | A^U   | F_1   |
| L7 Irish | Austral    | 88.87 | 86.94 | 76.29 | 81.27 | 89.61 | 94.69 | 92.08 |
|          | Boreal     | 97.06 | 98.78 | 96.53 | 97.64 | 94.31 | 97.96 | 96.10 |
|          | Mid.N.     | 93.90 | 97.04 | 89.02 | 92.86 | 91.74 | 97.82 | 94.68 |
|          | Mid.S.     | 97.19 | 98.96 | 95.94 | 97.43 | 95.13 | 98.75 | 96.91 |
|          | Polar.N.   | 83.55 | 81.99 | 84.55 | 83.25 | 85.11 | 82.61 | 83.84 |
|          | SubT.N.    | 98.16 | 99.44 | 98.65 | 99.04 | 69.09 | 84.38 | 75.97 |
|          | SubT.S.    | 95.35 | 99.02 | 95.10 | 97.02 | 83.51 | 96.34 | 89.47 |
|          | Tropical   | 89.72 | 86.48 | 42.76 | 57.23 | 90.00 | 98.72 | 94.16 |
|          | Mean       | 94.13 | 97.17 | 93.73 | 95.42 | 88.97 | 94.88 | 91.83 |
| L8 Biome | Barren     | 93.46 | 90.81 | 97.89 | 94.22 | 97.23 | 88.19 | 92.49 |
|          | Forest     | 96.65 | 90.65 | 88.18 | 89.40 | 97.76 | 98.26 | 98.01 |
|          | Grass      | 94.40 | 93.75 | 99.51 | 96.54 | 97.67 | 75.56 | 85.20 |
|          | Shrubland  | 99.24 | 98.69 | 99.83 | 99.26 | 99.83 | 98.64 | 99.23 |
|          | Snow       | 86.50 | 78.41 | 91.65 | 84.51 | 93.67 | 83.04 | 88.03 |
|          | Urban      | 95.29 | 88.87 | 98.29 | 93.34 | 99.08 | 93.76 | 96.35 |
|          | Wetlands   | 93.72 | 93.90 | 98.36 | 96.07 | 92.92 | 77.14 | 84.30 |
|          | Water      | 97.52 | 96.65 | 99.29 | 97.95 | 98.90 | 94.92 | 96.87 |
|          | Mean       | 94.35 | 91.75 | 97.75 | 94.65 | 97.47 | 90.79 | 94.01 |
| GF1 WHU  | Mean       | 94.46 | 92.07 | 97.62 | 94.76 | 97.31 | 91.11 | 94.11 |


### Quantitative evaluation for SCNN on L7 Irish, L8 Biome and GF1 WHU datasets

| Dataset  | Zone/Biome | Clear |       |       |       | Cloud |       |       |
|----------|------------|-------|-------|-------|-------|-------|-------|-------|
|          |            | A^O   | A^P   | A^U   | F_1   | A^P   | A^U   | F_1   |
| L7 Irish | Austral    | 88.87 | 86.94 | 76.29 | 81.27 | 89.61 | 94.69 | 92.08 |
|          | Boreal     | 97.06 | 98.78 | 96.53 | 97.64 | 94.31 | 97.96 | 96.10 |
|          | Mid.N.     | 93.90 | 97.04 | 89.02 | 92.86 | 91.74 | 97.82 | 94.68 |
|          | Mid.S.     | 97.19 | 98.96 | 95.94 | 97.43 | 95.13 | 98.75 | 96.91 |
|          | Polar.N.   | 83.55 | 81.99 | 84.55 | 83.25 | 85.11 | 82.61 | 83.84 |
|          | SubT.N.    | 98.16 | 99.44 | 98.65 | 99.04 | 69.09 | 84.38 | 75.97 |
|          | SubT.S.    | 95.35 | 99.02 | 95.10 | 97.02 | 83.51 | 96.34 | 89.47 |
|          | Tropical   | 89.72 | 86.48 | 42.76 | 57.23 | 90.00 | 98.72 | 94.16 |
|          | Mean       | 94.13 | 97.17 | 93.73 | 95.42 | 88.97 | 94.88 | 91.83 |
| L8 Biome | Barren     | 93.46 | 90.81 | 97.89 | 94.22 | 97.23 | 88.19 | 92.49 |
|          | Forest     | 96.65 | 90.65 | 88.18 | 89.40 | 97.76 | 98.26 | 98.01 |
|          | Grass      | 94.40 | 93.75 | 99.51 | 96.54 | 97.67 | 75.56 | 85.20 |
|          | Shrubland  | 99.24 | 98.69 | 99.83 | 99.26 | 99.83 | 98.64 | 99.23 |
|          | Snow       | 86.50 | 78.41 | 91.65 | 84.51 | 93.67 | 83.04 | 88.03 |
|          | Urban      | 95.29 | 88.87 | 98.29 | 93.34 | 99.08 | 93.76 | 96.35 |
|          | Wetlands   | 93.72 | 93.90 | 98.36 | 96.07 | 92.92 | 77.14 | 84.30 |
|          | Water      | 97.52 | 96.65 | 99.29 | 97.95 | 98.90 | 94.92 | 96.87 |
|          | Mean       | 94.35 | 91.75 | 97.75 | 94.65 | 97.47 | 90.79 | 94.01 |
| GF1 WHU  | Mean       | 94.46 | 92.07 | 97.62 | 94.76 | 97.31 | 91.11 | 94.11 |



## Citation

```bibtex
@InProceedings{LiJEI2024,
  author =    {Dengfeng Chai and Jingfeng Huang and Minghui Wu and 
               Xiaoping Yang and Ruisheng Wang},
  title =     {Remote sensing image cloud detection using a shallow convolutional neural network},
  booktitle = {ISPRS Journal of Photogrammetry and Remote Sensing},
  year =      {2024},
}
```