# CloudSEN12_High

> [CloudSEN12, a global dataset for semantic understanding of cloud and cloud shadow in Sentinel-2](https://www.nature.com/articles/s41597-022-01878-2)

## Introduction

- [Official Site](https://cloudsen12.github.io/download.html)
- [Paper Download](https://www.nature.com/articles/s41597-022-01878-2.pdf)
- Data Download: [Hugging Face](https://huggingface.co/datasets/csaybar/CloudSEN12-high)

## Abstract

Accurately characterizing clouds and their shadows is a long-standing problem in the Earth Observation community. Recent works showcase the necessity to improve cloud detection methods for imagery acquired by the Sentinel-2 satellites. However, the lack of consensus and transparency in existing reference datasets hampers the benchmarking of current cloud detection methods. Exploiting the analysis-ready data offered by the Copernicus program, we created CloudSEN12, a new multi-temporal global dataset to foster research in cloud and cloud shadow detection. CloudSEN12 has 49,400 image patches, including Sentinel-2 level-1C and level-2A multi-spectral data, Sentinel-1 synthetic aperture radar data, auxiliary remote sensing products, different hand-crafted annotations to label the presence of thick and thin clouds and cloud shadows, and  the results from eight state-of-the-art cloud detection algorithms. At present, CloudSEN12 exceeds all previous efforts in terms of annotation richness, scene variability, geographic distribution, metadata complexity, quality control, and number of samples.

## Dataset

CloudSEN12 is a LARGE dataset (~1 TB) for cloud semantic understanding that consists of 49,400 image patches (IP) that are evenly spread throughout all continents except Antarctica. Each IP covers 5090 x 5090 meters and contains data from Sentinel-2 levels 1C and 2A, hand-crafted annotations of thick and thin clouds and cloud shadows, Sentinel-1 Synthetic Aperture Radar (SAR), digital elevation model, surface water occurrence, land cover classes, and cloud mask results from six cutting-edge cloud detection algorithms.

![sample](https://github.com/user-attachments/assets/dd8d9b6c-b8ac-4b3f-bf8b-9c9954e9291f)

```
name: CloudSEN12
source: Sentinel-1,2
band: 12
resolution: 10m
pixel: 512x512
train: 8490
val: 535
test: 975
disk: (~1 TB)
annotation: 
    - 0: Clear
    - 1: Thick cloud
    - 2: Thin cloud
    - 3: Cloud shadow
scene: -
```

## Citation

```
@article{cloudsen12,
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
