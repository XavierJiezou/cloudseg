# CDnetV2: CNN-Based Cloud Detection for Remote Sensing Imagery With Cloud-Snow Coexistence

> [CDnetV2: CNN-Based Cloud Detection for Remote Sensing Imagery With Cloud-Snow Coexistence](https://ieeexplore.ieee.org/document/9094671)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://ieeexplore.ieee.org/document/9094671">Official Repo</a>

<a href="https://github.com/nkszjx/CDnetV2-pytorch-master">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

Cloud detection is a crucial preprocessing step for optical satellite remote sensing (RS) images. This article focuses on the cloud detection for RS imagery with cloud-snow coexistence and the utilization of the satellite thumbnails that lose considerable amount of high resolution and spectrum information of original RS images to extract cloud mask efficiently. To tackle this problem, we propose a novel cloud detection neural network with an encoder-decoder structure, named CDnetV2, as a series work on cloud detection. Compared with our previous CDnetV1, CDnetV2 contains two novel modules, that is, adaptive feature fusing model (AFFM) and high-level semantic information guidance flows (HSIGFs). AFFM is used to fuse multilevel feature maps by three submodules: channel attention fusion model (CAFM), spatial attention fusion model (SAFM), and channel attention refinement model (CARM). HSIGFs are designed to make feature layers at decoder of CDnetV2 be aware of the locations of the cloud objects. The high-level semantic information of HSIGFs is extracted by a proposed high-level feature fusing model (HFFM). By being equipped with these two proposed key modules, AFFM and HSIGFs, CDnetV2 is able to fully utilize features extracted from encoder layers and yield accurate cloud detection results. Experimental results on the ZY-3 satellite thumbnail data set demonstrate that the proposed CDnetV2 achieves accurate detection accuracy and outperforms several state-of-the-art methods.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/user-attachments/assets/8e213e6f-3c7a-45f1-b9fc-de7fe6c92489" width="70%"/>
</div>

## Results and models

### CLOUD EXTRACTION ACCURACY (%) OF DIFFERENT CNN-BASED METHODS ON ZY-3 SATELLITE THUMBNAILS


| Method | OA    | MIoU  | Kappa | PA    | UA    |
|-------|-------|-------|-------|-------|-------|
| MSegNet | 90.86 | 81.20 | 75.57 | 73.78 | 86.13 |
| MUnet | 91.62 | 82.51 | 76.70 | 74.44 | 87.39 |
| PSPnet | 90.58 | 81.63 | 75.36 | 76.02 | 87.52 |
| DeeplabV3+ | 91.80 | 82.62 | 77.65 | 75.30 | 87.76 |
| CDnetV1 | 93.15 | 82.80 | 79.21 | 82.37 | 86.72 |
| CDnetV2 | 95.76 | 86.62 | 82.51 | 87.75 | 88.58 |



### STATISTICAL RESULTS OF CLOUDAGE ESTIMATION ERROR IN TERMS OF THE MAD AND ITS VARIANCE


| Methods    | Mean value ($\mu$) | Standard Deviation ($\sigma^2$)) |
|------------|--------------------|----------------------------------|
| CDnetV2    | 0.0241             | 0.0220                           |
| CDnetV1    | 0.0357             | 0.0288                           |
| DeeplabV3+ | 0.0456             | 0.0301                           |
| PSPnet     | 0.0487             | 0.0380                           |
| MUnet      | 0.0544             | 0.0583                           |
| MSegNet    | 0.0572             | 0.0591                           |




### COMPUTATIONAL COMPLEXITY ANALYS IS OF DIFFERENT CNN-BASED METHODS

| Methods    | GFLOPs(224×224) | Trainable params | Running time (s)(1k×1k) |
|------------|-----------------|------------------|-------------------------|
| CDnetV2    | 31.5            | 65.9 M           | 1.31                    |
| CDnetV1    | 48.5            | 64.8 M           | 1.26                    |
| DeeplabV3+ | 31.8            | 40.3 M           | 1.14                    |
| PSPnet     | 19.3            | 46.6 M           | 1.05                    |
| MUnet      | 25.2            | 8.6 M            | 1.09                    |
| MSegNet    | 90.2            | 29.7 M           | 1.28                    |




## Citation

```bibtex
@ARTICLE{8681238,
author={J. {Yang} and J. {Guo} and H. {Yue} and Z. {Liu} and H. {Hu} and K. {Li}},
journal={IEEE Transactions on Geoscience and Remote Sensing},
title={CDnet: CNN-Based Cloud Detection for Remote Sensing Imagery},
year={2019}, volume={57},
number={8}, pages={6195-6211},
doi={10.1109/TGRS.2019.2904868} }

@ARTICLE{9094671,
author={J. {Guo} and J. {Yang} and H. {Yue} and H. {Tan} and C. {Hou} and K. {Li}},
journal={IEEE Transactions on Geoscience and Remote Sensing},
title={CDnetV2: CNN-Based Cloud Detection for Remote Sensing Imagery With Cloud-Snow Coexistence},
year={2021},
volume={59},
number={1},
pages={700-713},
doi={10.1109/TGRS.2020.2991398} }
```
