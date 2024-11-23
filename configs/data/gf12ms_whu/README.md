# GF12MS_WHU

> [Transferring Deep Models for Cloud Detection in Multisensor Images via Weakly Supervised Learning](https://ieeexplore.ieee.org/document/10436637)

## Introduction

- [Official Site](https://github.com/whu-ZSC/GF1-GF2MS-WHU)
- [Paper Download](https://zhiweili.net/assets/pdf/2024.2_TGRS_Transferring%20Deep%20Models%20for%20Cloud%20Detection%20in%20Multisensor%20Images%20via%20Weakly%20Supervised%20Learning.pdf)
- Data Download: [Baidu Disk](https://pan.baidu.com/s/1kBpym0mW_TS9YL1GQ9t8Hw) (password: 9zuf)

## Abstract

Recently, deep learning has been widely used for cloud detection in satellite images; however, due to radiometric and spatial resolution differences in images from different sensors and time-consuming process of manually labeling cloud detection datasets, it is difficult to effectively generalize deep learning models for cloud detection in multisensor images. This article propose a weakly supervised learning method for transferring deep models for cloud detection in multisensor images (TransMCD), which leverages the generalization of deep models and the spectral features of clouds to construct pseudo-label dataset to improve the generalization of models. A deep model is first pretrained using a well-annotated cloud detection dataset, which is used to obtain a rough cloud mask of unlabeled target image. The rough mask can be used to determine the spectral threshold adaptively for cloud segmentation of target image. Block-level pseudo labels with high confidence in target image are selected using the rough mask and spectral mask. Unsupervised segmentation technique is used to construct a high-quality pixel-level pseudo-label dataset. Finally, the pseudo-label dataset is used as supervised information for transferring the pretrained model to target image. The TransMCD method was validated by transferring model trained on 16-m Gaofen-1 wide field of view(WFV)images to 8-m Gaofen-1, 4-m Gaofen-2, and 10-m Sentinel-2 images. The F1-score of the transferred models on target images achieves improvements of 1.23%–9.63% over the pretrained models, which is comparable to the fully-supervised models trained with well-annotated target images, suggesting the efficiency of the TransMCD method for cloud detection in multisensor images.

## Dataset

### GF1MS-WHU Dataset

> The two GF-1 PMS sensors have four MS bands with an 8-m spatial resolution and a panchromatic (PAN) band with a higher spatial resolution of 2 m. The spectral range of the MS bands is identical to that of the WFV sensors. In this study, 141 unlabeled images collected from various regions in China were used as the training data for the proposed method. In addition, 33 labeled images were used as the training data for the fully supervised methods, as well as the validation data for the different methods. The acquisition of the images spanned from June 2014 to December 2020 and encompassed four MS bands in both PMS sensors. Note that Fig. 7 only presents the distribution regions of the labeled images. 

```yaml
name: GF1MS-WHU
source: GaoFen-1
band: 4 (MS)
resolution: 8m (MS), 2m (PAN)
pixel: 250x250
train: 6343
val: -
test: 4085
disk: 10.8GB
annotation:
  - 0: clear sky
  - 1: cloud
scene: [Forest,Urban,Barren,Water,Farmland,Grass,Wetland]
```

### GF2MS-WHU Dataset

> The GF-2 satellite is configured with two PMS sensors. Each sensor has four MS bands with a 4-m spatial resolution and a PAN band with a 1-m spatial resolution. The GF-2 PMS sensors have the same bandwidth as the GF-1 WFV sensors. In this study, 163 unlabeled images obtained from Hubei, Jilin, and Hainan provinces were used as the training data for the proposed method, and 29 labeled images were used as the training data for the fully supervised methods, as well as the validation data for the different methods. The images were acquired from June 2014 to October 2020 and included four MS bands in both PMS sensors.

```yaml
name: GF2MS-WHU
source: GaoFen-2
band: 4 (MS)
resolution: 4m (MS), 1m (PAN)
pixel: 250x250
train: 14357
val: -
test: 7560
disk: 26.7GB
annotation:
  - 0: clear sky
  - 1: cloud
scene: [Forest,Urban,Barren,Water,Farmland,Grass,Wetland]
```


## Citation

```bibtex
@ARTICLE{gaofen12,
  author={Zhu, Shaocong and Li, Zhiwei and Shen, Huanfeng},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Transferring Deep Models for Cloud Detection in Multisensor Images via Weakly Supervised Learning}, 
  year={2024},
  volume={62},
  number={},
  pages={1-18},
  keywords={Cloud computing;Clouds;Sensors;Predictive models;Supervised learning;Image segmentation;Deep learning;Cloud detection;deep learning;multisensor images;weakly supervised learning},
  doi={10.1109/TGRS.2024.3358824}
}
```
