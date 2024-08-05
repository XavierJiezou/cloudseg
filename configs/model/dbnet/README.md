# Dual-Branch Network for Cloud and Cloud Shadow Segmentation


> [Dual-Branch Network for Cloud and Cloud Shadow Segmentation](https://ieeexplore.ieee.org/document/9775689)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://ieeexplore.ieee.org/document/9775689">Official Repo</a>

<a href="https://github.com/Beyond0128/Dual-Branch-network-for-segmentation/tree/main">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

— Cloud and cloud shadow segmentation is one of the
most important issues in remote sensing image processing. Most
of the remote sensing images are very complicated. In this work,
a dual-branch model composed of transformer and convolution
network is proposed to extract semantic and spatial detail information of the image, respectively, to solve the problems of false
detection and missed detection. To improve the model’s feature
extraction, a mutual guidance module (MGM) is introduced,
so that the transformer branch and the convolution branch can
guide each other for feature mining. Finally, in view of the problem of rough segmentation boundary, this work uses different
features extracted by the transformer branch and the convolution
branch for decoding and repairs the rough segmentation boundary in the decoding part to make the segmentation boundary
clearer. Experimental results on the Landsat-8, Sentinel-2 data,
the public dataset high-resolution cloud cover validation dataset
created by researchers at Wuhan University (HRC_WHU), and
the public dataset Spatial Procedures for Automated Removal
of Cloud and Shadow (SPARCS) demonstrate the effectiveness
of our method and its superiority to the existing state-of-the-art
cloud and cloud shadow segmentation approaches.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/user-attachments/assets/7fcfa82c-bb50-4788-851c-4bf62bbf5903" width="70%"/>
</div>

## Results and models

### COMPARISON OF EVALUATION METRICS OF DIFFERENT MODELS ON CLOUD AND CLOUD SHADOW DATASET


| **Method**         | **Cloud** |          |          |           | **Cloud Shadow** |          |          |           |           |            |             |
|--------------------|-----------|----------|----------|-----------|------------------|----------|----------|-----------|-----------|------------|-------------|
|                    | **OA(%)** | **P(%)** | **R(%)** | **F₁(%)** | **OA(%)**        | **P(%)** | **R(%)** | **F₁(%)** | **PA(%)** | **MPA(%)** | **MIoU(%)** |
| FCN-8S [7]         | 95.87     | 88.47    | 92.8     | 90.63     | 97.19            | 86.87    | 88.72    | 87.79     | 93.40     | 90.52      | 84.01       |
| PAN [38]           | 98.25     | 96.29    | 95.49    | 95.89     | 98.31            | 92.71    | 92.46    | 92.59     | 96.73     | 95.52      | 91.26       |
| BiseNet V2 [35]    | 98.27     | 96.57    | 95.28    | 95.92     | 98.34            | 94.18    | 90.99    | 92.59     | 96.68     | 95.96      | 91.20       |
| PSPNet [11]        | 98.35     | 96.13    | 96.13    | 96.13     | 98.40            | 92.70    | 93.27    | 92.99     | 96.87     | 95.55      | 91.69       |
| DeepLab V3Plus [9] | 98.65     | 97.70    | 95.94    | 96.82     | 98.66            | 93.88    | 94.32    | 94.10     | 97.37     | 96.48      | 92.99       |
| LinkNet [39]       | 98.61     | 96.59    | 96.91    | 96.75     | 98.54            | 94.19    | 92.91    | 93.55     | 97.23     | 96.24      | 92.55       |
| ExtremeC3Net [40]  | 98.64     | 97.32    | 96.28    | 96.80     | 98.60            | 94.68    | 92.95    | 93.82     | 97.30     | 96.57      | 92.76       |
| DANet [41]         | 96.45     | 91.68    | 91.72    | 91.71     | 97.29            | 88.40    | 87.68    | 88.04     | 94.03     | 91.93      | 85.07       |
| CGNet [42]         | 98.37     | 95.93    | 96.48    | 96.20     | 98.27            | 93.33    | 91.34    | 92.34     | 96.73     | 95.60      | 91.27       |
| PVT [23]           | 98.57     | 97.45    | 95.84    | 96.65     | 98.55            | 93.28    | 94.08    | 93.68     | 97.21     | 96.18      | 92.55       |
| CvT [24]           | 98.44     | 95.89    | 96.88    | 96.38     | 98.32            | 92.90    | 92.24    | 92.57     | 96.85     | 95.54      | 91.57       |
| modified VGG [12]  | 98.40     | 98.13    | 94.30    | 96.22     | 98.57            | 94.41    | 92.88    | 93.64     | 97.04     | 96.56      | 92.17       |
| CloudNet [13]      | 98.70     | 97.22    | 96.68    | 96.95     | 98.40            | 92.05    | 94.05    | 93.05     | 97.17     | 95.77      | 92.36       |
| GAFFNet [43]       | 98.53     | 96.49    | 96.63    | 96.56     | 98.41            | 92.71    | 93.40    | 93.05     | 97.06     | 95.73      | 92.08       |
| Our                | 98.76     | 97.95    | 96.22    | 97.08     | 98.73            | 94.39    | 94.39    | 94.39     | 97.56     | 96.77      | 93.42       |

### COMPARISON OF EVALUATION METRICS OF DIFFERENT MODELS ON THE SPARCS DATASET


| Method | Class Pixel Accuracy |  |  |  |  | Overall Results |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|
| | Cloud(%) | Cloud Shadow(%) | Snow/Ice(%) | Water(%) | Land(%) | PA(%) | Recall(%) | Precision(%) | F₁(%) | MIoU(%) |
| PAN [38] | 89.10 | 75.27 | 86.60 | 79.96 | 95.64 | 91.20 | 87.34 | 85.32 | 81.53 | 76.57 |
| BiSeNet V2 [35] | 85.87 | 64.75 | 93.84 | 81.44 | 97.17 | 91.31 | 89.77 | 84.61 | 83.09 | 77.79 |
| PSPNet [11] | 90.79 | 63.75 | 94.22 | 77.73 | 96.84 | 91.78 | 90.29 | 84.67 | 83.48 | 78.20 |
| DeepLab V3Plus [9] | 87.81 | 72.12 | 85.17 | 81.27 | 97.84 | 91.99 | 90.75 | 84.85 | 84.01 | 78.44 |
| LinkNet [39] | 85.35 | 74.38 | 91.92 | 80.30 | 96.44 | 91.31 | 88.66 | 85.68 | 82.81 | 77.87 |
| ExtremeC3Net [40] | 91.09 | 75.47 | 95.43 | 83.62 | 96.13 | 92.77 | 90.32 | 88.35 | 85.46 | 81.29 |
| DANet [41] | 82.06 | 42.25 | 91.28 | 73.65 | 95.03 | 86.92 | 83.86 | 76.85 | 74.64 | 68.33 |
| CGNet [42] | 90.63 | 72.78 | 95.37 | 83.30 | 96.51 | 93.22 | 91.00 | 88.95 | 86.30 | 82.28 |
| PVT [23] | 88.22 | 75.77 | 92.00 | 86.27 | 95.92 | 92.02 | 89.76 | 87.64 | 84.66 | 80.24 |
| CvT [24] | 88.24 | 71.63 | 95.41 | 87.71 | 96.14 | 92.17 | 89.83 | 87.83 | 84.80 | 80.55 |
| modified VGG [12] | 85.55 | 58.53 | 94.87 | 79.35 | 95.98 | 89.99 | 86.38 | 82.85 | 79.36 | 74.00 |
| CloudNet [13] | 85.99 | 74.58 | 91.78 | 80.34 | 96.52 | 91.50 | 88.49 | 85.84 | 82.79 | 77.95 |
| GAFFNet [43] | 86.97 | 59.00 | 85.21 | 78.06 | 94.47 | 88.62 | 86.70 | 80.74 | 78.56 | 72.32 |
| our | 91.12 | 78.38 | 96.59 | 89.99 | 97.52 | 94.31 | 92.90 | 90.72 | 88.83 | 85.26 |






## Citation

```bibtex
@ARTICLE{9775689,
  author={Lu, Chen and Xia, Min and Qian, Ming and Chen, Binyu},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Dual-Branch Network for Cloud and Cloud Shadow Segmentation}, 
  year={2022},
  volume={60},
  number={},
  pages={1-12},
  keywords={Feature extraction;Transformers;Convolution;Clouds;Image segmentation;Decoding;Task analysis;Deep learning;dual branch;remote sensing image;segmentation},
  doi={10.1109/TGRS.2022.3175613}}

```
