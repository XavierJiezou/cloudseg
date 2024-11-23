# UNetMobV2

> [CloudSEN12, a global dataset for semantic understanding of cloud and cloud shadow in Sentinel-2](https://www.nature.com/articles/s41597-022-01878-2#Tab5)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/cloudsen12/models">Official Repo</a>

## Abstract

<!-- [ABSTRACT] -->

U-Net models often have considerable memory requirements since the encoder and decoder components include skip connections of large tensors. However, the MobileNetV2 encoder significantly decreases memory utilization due to the use of depthwise separable convolutions and inverted residuals. The entire memory requirements of our model, considering a batch with a single image (1 × 13 × 512 × 512), the forward/backward pass, and model parameters, is less than 1 GB using the PyTorch deep learning library61. The implementation of the proposed model can be found at https://github.com/cloudsen12/models.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/user-attachments/assets/734d7f11-631e-4460-99d1-db7e4e8766c5" width="70%"/>
</div>

## Results and models

### Metrics of the three different experiments for all the annotation algorithms.

| **Experiment**        | **CD algorithm** | **BOA** | **PAlow%** | **PAmiddle%** | **PAhigh%** | **UAlow%** | **UAmiddle%** | **UAhigh%** |
|-----------------------|------------------|---------|------------|---------------|-------------|------------|---------------|-------------|
| **a. cloud/no cloud** | Human level      | 0.99    | 1.03       | 14.03         | 84.94       | 0.13       | 4.39          | 95.48       |
|                       | UNetMobV2        | 0.92    | 0.77       | 30.63         | 68.6        | 0.26       | 25.03         | 74.71       |
|                       | KappaMask L2A    | 0.77    | 2.83       | 31.92         | 65.25       | 1.56       | 63.04         | 35.41       |
|                       | KappaMask L1C    | 0.82    | 4.89       | 45.3          | 49.81       | 0.65       | 38.38         | 60.97       |
|                       | Fmask            | 0.84    | 5.92       | 40.54         | 53.54       | 0.26       | 52.65         | 47.09       |
|                       | s2cloudless      | 0.79    | 7.08       | 52.38         | 40.54       | 0.65       | 31.5          | 67.84       |
|                       | Sen2Cor          | 0.71    | 13.13      | 64.86         | 22.01       | 1.58       | 20.05         | 78.36       |
|                       | QA60             | 0.58    | 24.84      | 49.94         | 25.23       | 1.39       | 37.62         | 60.99       |
|                       | CD-FCNN-RGBI     | 0.72    | 17.50      | 74.00         | 8.49        | 1.62       | 12.58         | 85.79       |
|                       | CD-FCNN-RGBISWIR | 0.72    | 18.40      | 71.43         | 10.17       | 0.82       | 9.43          | 89.75       |
| **b. cloud shadow**   | Human level      | 0.99    | 3.11       | 22.04         | 74.85       | 0.60       | 9.97          | 89.43       |
|                       | UNetMobV2        | 0.89    | 8.88       | 67.16         | 23.96       | 7.99       | 46.65         | 45.36       |
|                       | KappaMask L2A    | 0.64    | 37.28      | 59.76         | 2.96        | 12.24      | 36.9          | 50.85       |
|                       | KappaMask L1C    | 0.74    | 30.03      | 60.95         | 9.02        | 20.67      | 59.36         | 19.97       |
|                       | Fmask            | 0.72    | 22.34      | 76.04         | 1.63        | 14.53      | 77.06         | 8.41        |
|                       | Sen2Cor          | 0.51    | 64.5       | 35.21         | 0.30        | 6.90       | 18.10         | 75.00       |
| **c. valid/invalid**  | Human level      | 0.99    | 1.03       | 14.8          | 84.17       | 0.13       | 2.33          | 97.55       |
|                       | UNetMobV2        | 0.91    | 0.77       | 28.57         | 70.66       | 0.00       | 17.14         | 82.86       |
|                       | KappaMask L2A    | 0.75    | 2.96       | 39.38         | 57.66       | 1.29       | 44.32         | 54.39       |
|                       | KappaMask L1C    | 0.81    | 3.99       | 47.62         | 48.39       | 0.65       | 32.64         | 66.71       |
|                       | Fmask            | 0.81    | 4.89       | 45.43         | 49.68       | 0.26       | 44.34         | 55.39       |
|                       | Sen2Cor          | 0.67    | 13.77      | 69.63         | 16.6        | 1.05       | 18.58         | 80.37       |


## Citation

```bibtex
@article{aybar2022cloudsen12,
  title={CloudSEN12, a global dataset for semantic understanding of cloud and cloud shadow in Sentinel-2},
  author={Aybar, Cesar and Ysuhuaylas, Luis and Loja, Jhomira and Gonzales, Karen and Herrera, Fernando and Bautista, Lesly and Yali, Roy and Flores, Angie and Diaz, Lissette and Cuenca, Nicole and others},
  journal={Scientific data},
  volume={9},
  number={1},
  pages={782},
  year={2022},
  publisher={Nature Publishing Group UK London}
}
```
