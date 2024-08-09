# MCDNet: Multilevel cloud detection network for remote sensing images based on dual-perspective change-guided and multi-scale feature fusion

> [MCDNet: Multilevel cloud detection network for remote sensing images based on dual-perspective change-guided and multi-scale feature fusion](https://www.sciencedirect.com/science/article/pii/S1569843224001742?via%3Dihub)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/djw-easy/MCDNet">Official Repo</a>

## Abstract

<!-- [ABSTRACT] -->
Cloud detection plays a crucial role in the preprocessing of optical remote sensing images. While extensive deep learning-based methods have shown strong performance in detecting thick clouds, their ability to identify thin and broken clouds is often inadequate due to their sparse distribution, semi-transparency, and similarity to background regions. To address this limitation, we introduce a multilevel cloud detection network (MCDNet) capable of simultaneously detecting thick and thin clouds. This network effectively enhances the accuracy of identifying thin and broken clouds by integrating a dual-perspective change-guided mechanism (DPCG) and a multi-scale feature fusion module (MSFF). The DPCG creates a dual-input stream by combining the original image with the thin cloud removal image, and then utilizes a dual-perspective feature fusion module (DPFF) to perform feature fusion and extract change features, thereby improving the model's ability to perceive thin cloud regions and mitigate inter-class similarity in multilevel cloud detection. The MSFF enhances the model's sensitivity to broken clouds by utilizing multiple non-adjacent low-level features to remedy the missing spatial information in the high-level features during multiple downsampling. Experimental results on the L8-Biome and WHUS2-CD datasets demonstrate that MCDNet significantly enhances the detection performance of both thin and broken clouds, and outperforms state-of-the-art methods in accuracy and efficiency. 

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/user-attachments/assets/dd0a6a20-da10-4f8e-949b-3722af001b28" width="70%"/>
</div>


## Results and models

### Quantitative performance of MCDNet with different thin cloud removal methods on the L8-Biome dataset.


| Thin cloud removal method | OA   | IoU  | Specificity | Kappa |
|--------------------------|------|------|-------------|--------|
| DCP                      | 94.18| 89.55| 97.09       | 70.42  |
| HF                       | 94.22| 89.69| 97.11       | 70.59  |
| BCCR                     | 94.25| 89.75| 97.13       | 71.01  |

###  Quantitative performance of MCDNet with different components using BCCR thin cloud removal method on the L8-Biome dataset


| Baseline       | √     | √     | √     | √     |
|----------------|-------|-------|-------|-------|
| MSFF           | ×     | √     | ×     | √     |
| DPFF           | ×     | ×     | √     | √     |
| OA             | 92.91 | 93.25 | 93.81 | 94.25 |
| IoU            | 87.58 | 88.13 | 88.99 | 89.75 |
| Specificity    | 96.46 | 96.62 | 96.90 | 97.13 |
| Kappa          | 67.22 | 68.88 | 69.94 | 71.01 |
| Parameters (M) | 8.89  | 10.67 | 11.34 | 13.11 |



### Quantitative comparisons of different methods on the L8-Biome dataset.


| Method   | OA    | IoU   | Specificity | Kappa |
|----------|-------|-------|-------------|-------|
| UNet     | 90.78 | 84.44 | 95.39       | 63.22 |
| SegNet   | 91.44 | 85.33 | 95.72       | 63.98 |
| HRNet    | 91.17 | 84.87 | 95.59       | 62.32 |
| SwinUnet | 91.73 | 85.74 | 95.86       | 64.36 |
| MFCNN    | 92.39 | 86.76 | 96.20       | 66.72 |
| MSCFF    | 92.49 | 86.99 | 96.25       | 65.82 |
| CDNetV2  | 91.55 | 85.51 | 95.77       | 64.91 |
| CloudNet | 92.19 | 86.55 | 96.09       | 65.99 |
| MCDNet   | 94.25 | 89.75 | 97.13       | 71.01 |


### Quantitative comparisons of different methods on the WHUS2-CD dataset.

| Method   | OA    | IoU   | Specificity | Kappa |
|----------|-------|-------|-------------|-------|
| UNet     | 98.24 | 63.72 | 99.51       | 63.45 |
| SegNet   | 98.09 | 62.53 | 99.15       | 61.71 |
| HRNet    | 97.91 | 61.73 | 99.41       | 59.29 |
| SwinUnet | 98.69 | 61.93 | 99.48       | 67.81 |
| MFCNN    | 98.56 | 64.79 | 98.92       | 65.52 |
| MSCFF    | 98.84 | 66.21 | 99.59       | 68.73 |
| CDNetV2  | 98.68 | 65.69 | 99.54       | 67.21 |
| CloudNet | 98.58 | 65.82 | 99.55       | 68.57 |
| MCDNet   | 98.97 | 66.45 | 99.58       | 69.42 |


### Quantitative performance of different fusion schemes for extracting change features on L8-Biome dataset.

| Methods | Fusion schemes | OA    | IoU   | Specificity | Kappa | Params (M) |
|---------|----------------|-------|-------|-------------|-------|------------|
| (a)     | no fusion      | 93.25 | 88.13 | 96.62       | 68.88 | 10.67      |
| (b)     | Concatenation  | 94.14 | 89.62 | 97.07       | 66.49 | 11.36      |
| (c)     | Subtraction    | 91.05 | 84.85 | 95.52       | 64.32 | 10.67      |
| (d)     | CDM            | 93.77 | 89.02 | 96.88       | 65.46 | 12.07      |
| (e)     | DPFF           | 94.25 | 89.75 | 97.13       | 71.01 | 13.11      |


### Quantitative performance of different multi-scale feature fusion schemes on L8-Biome dataset.

| Methods   | Fusion schemes | OA   | IoU  | Specificity | Kappa | Params (M) |
|-----------|----------------|------|------|-------------|--------|------------|
| (a)       | no fusion      | 93.81| 88.99| 96.90       | 69.94  | 11.34      |
| (b)       | HRNet          | 94.03| 89.47| 97.02       | 69.59  | 12.58      |
| (c)       | MSCFF          | 93.45| 88.45| 96.72       | 68.87  | 14.86      |
| (d)       | CloudNet       | 93.92| 89.29| 96.96       | 66.97  | 11.34      |
| (e)       | MSFF           | 94.25| 89.75| 97.13       | 71.01  | 13.11      |

## Citation

```bibtex
@article{MCDNet,
title = {MCDNet: Multilevel cloud detection network for remote sensing images based on dual-perspective change-guided and multi-scale feature fusion},
journal = {International Journal of Applied Earth Observation and Geoinformation},
volume = {129},
pages = {103820},
year = {2024},
issn = {1569-8432},
doi = {10.1016/j.jag.2024.103820}
}
```
