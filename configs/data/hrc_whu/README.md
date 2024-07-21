# HRC_WHU

> [Deep learning based cloud detection for medium and high resolution remote sensing images of different sensors](https://www.sciencedirect.com/science/article/pii/S0924271619300565)

## Introduction

<!-- [ALGORITHM] -->

<a href="http://sendimage.whu.edu.cn/en/hrc_whu/">Official Site</a>

## Abstract

<!-- [ABSTRACT] -->

Cloud detection is an important preprocessing step for the precise application of optical satellite imagery. In this paper, we propose a deep learning based cloud detection method named multi-scale convolutional feature fusion (MSCFF) for remote sensing images of different sensors. In the network architecture of MSCFF, the symmetric encoder-decoder module, which provides both local and global context by densifying feature maps with trainable convolutional filter banks, is utilized to extract multi-scale and high-level spatial features. The feature maps of multiple scales are then up-sampled and concatenated, and a novel multi-scale feature fusion module is designed to fuse the features of different scales for the output. The two output feature maps of the network are cloud and cloud shadow maps, which are in turn fed to binary classifiers outside the model to obtain the final cloud and cloud shadow mask. The MSCFF method was validated on hundreds of globally distributed optical satellite images, with spatial resolutions ranging from 0.5 to 50 m, including Landsat-5/7/8, Gaofen-1/2/4, Sentinel-2, Ziyuan-3, CBERS-04, Huanjing-1, and collected high-resolution images exported from Google Earth. The experimental results show that MSCFF achieves a higher accuracy than the traditional rule-based cloud detection methods and the state-of-the-art deep learning models, especially in bright surface covered areas. The effectiveness of MSCFF means that it has great promise for the practical application of cloud detection for multiple types of medium and high-resolution remote sensing images. Our established global high-resolution cloud detection validation dataset has been made available online (http://sendimage.whu.edu.cn/en/mscff/).

<!-- [IMAGE] -->

<div align=center>
<img src="http://sendimage.whu.edu.cn/en/wp-content/uploads/2018/11/Preview-of-HRC_WHU.jpg" width="800"/>
</div>

## Dataset

```yaml
name: hrc_whu
source: google earth
band: 3 (rgb)
resolution: 0.5m-15m
pixel: 1280x720
train: 120
val: null
test: 30
disk: 168mb
annotation: 
    - 0: clear sky
    - 1: cloud
scene: [water, vegetation, urban, snow/ice, barren]
```

## Citation

```bibtex
@article{hrc_whu,
title = {Deep learning based cloud detection for medium and high resolution remote sensing images of different sensors},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {150},
pages = {197-212},
year = {2019},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2019.02.017},
url = {https://www.sciencedirect.com/science/article/pii/S0924271619300565},
author = {Zhiwei Li and Huanfeng Shen and Qing Cheng and Yuhao Liu and Shucheng You and Zongyi He},
keywords = {Cloud detection, Cloud shadow, Convolutional neural network, Multi-scale, Convolutional feature fusion, MSCFF}
}
```