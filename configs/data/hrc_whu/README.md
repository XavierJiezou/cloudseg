# HRC_WHU

> [Deep learning based cloud detection for medium and high resolution remote sensing images of different sensors](https://www.sciencedirect.com/science/article/pii/S0924271619300565)

## Introduction

- [Official Site](http://sendimage.whu.edu.cn/en/hrc_whu/)
- [Paper Download](http://sendimage.whu.edu.cn/en/wp-content/uploads/2019/03/2019_PHOTO_Zhiwei-Li_Deep-learning-based-cloud-detection-for-medium-and-high-resolution-remote-sensing-images-of-different-sensors.pdf)
- Data Download: [Baidu Disk](https://pan.baidu.com/s/1thOTKVO2iTAalFAjFI2_ZQ) (password: ihfb) or [Google Drive](https://drive.google.com/file/d/1qqikjaX7tkfOONsF5EtR4vl6J7sToA6p/view?usp=sharing)

## Abstract

Cloud detection is an important preprocessing step for the precise application of optical satellite imagery. In this paper, we propose a deep learning based cloud detection method named multi-scale convolutional feature fusion (MSCFF) for remote sensing images of different sensors. In the network architecture of MSCFF, the symmetric encoder-decoder module, which provides both local and global context by densifying feature maps with trainable convolutional filter banks, is utilized to extract multi-scale and high-level spatial features. The feature maps of multiple scales are then up-sampled and concatenated, and a novel multi-scale feature fusion module is designed to fuse the features of different scales for the output. The two output feature maps of the network are cloud and cloud shadow maps, which are in turn fed to binary classifiers outside the model to obtain the final cloud and cloud shadow mask. The MSCFF method was validated on hundreds of globally distributed optical satellite images, with spatial resolutions ranging from 0.5 to 50 m, including Landsat-5/7/8, Gaofen-1/2/4, Sentinel-2, Ziyuan-3, CBERS-04, Huanjing-1, and collected high-resolution images exported from Google Earth. The experimental results show that MSCFF achieves a higher accuracy than the traditional rule-based cloud detection methods and the state-of-the-art deep learning models, especially in bright surface covered areas. The effectiveness of MSCFF means that it has great promise for the practical application of cloud detection for multiple types of medium and high-resolution remote sensing images. Our established global high-resolution cloud detection validation dataset has been made available online (http://sendimage.whu.edu.cn/en/mscff/).

## Dataset

The high-resolution cloud cover validation dataset was created by the SENDIMAGE Lab in Wuhan University, and has been termed HRC_WHU. The HRC_WHU data comprise 150 high-resolution images acquired with three RGB channels and a resolution varying from 0.5 to 15 m in different global regions. As shown in Fig. 1, the images were collected from Google Earth (Google Inc.) in five main land-cover types, i.e., water, vegetation, urban, snow/ice, and barren. The associated reference cloud masks were digitized by experts in the field of remote sensing image interpretation. The established high-resolution cloud cover validation dataset has been made available online.

![sample](https://github.com/user-attachments/assets/16ca8da2-7e54-49df-9d17-2420a8531cd0)

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

## Annotation

In the procedure of delineating the cloud mask for high-resolution imagery, we first stretched the cloudy image to the appropriate contrast in Adobe Photoshop. The lasso tool and magic wand tool were then alternately used to mark the locations of the clouds in the image. The manually labeled reference mask was finally created by assigning the pixel values of cloud and clear sky to 255 and 0, respectively. Note that a tolerance of 5–30 was set when using the magic wand tool, and the lasso tool was used to modify the areas that could not be correctly selected by the magic wand tool. As we did in a previous study (Li et al., 2017), the thin clouds were labeled as cloud if they were visually identifiable and the underlying surface could not be seen clearly. Considering that cloud shadows in high-resolution images are rare and hard to accurately select, only clouds were labeled in the reference masks.

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
