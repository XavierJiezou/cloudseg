# L8-Biome

> [Cloud detection algorithm comparison and validation for operational Landsat data products](https://www.sciencedirect.com/science/article/abs/pii/S0034425717301293)

## Introduction

- [Official Site](https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data)
- [Paper Download](https://gerslab.cahnr.uconn.edu/wp-content/uploads/sites/2514/2021/06/1-s2.0-S0034425717301293-Steve_Foga_cloud_detection_2017.pdf)
- Data Download: [USGS](https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data)

## Abstract

Clouds are a pervasive and unavoidable issue in satellite-borne optical imagery. Accurate, well-documented, and automated cloud detection algorithms are necessary to effectively leverage large collections of remotely sensed data. The Landsat project is uniquely suited for comparative validation of cloud assessment algorithms because the modular architecture of the Landsat ground system allows for quick evaluation of new code, and because Landsat has the most comprehensive manual truth masks of any current satellite data archive. Currently, the Landsat Level-1 Product Generation System (LPGS) uses separate algorithms for determining clouds, cirrus clouds, and snow and/or ice probability on a per-pixel basis. With more bands onboard the Landsat 8 Operational Land Imager (OLI)/Thermal Infrared Sensor (TIRS) satellite, and a greater number of cloud masking algorithms, the U.S. Geological Survey (USGS) is replacing the current cloud masking workflow with a more robust algorithm that is capable of working across multiple Landsat sensors with minimal modification. Because of the inherent error from stray light and intermittent data availability of TIRS, these algorithms need to operate both with and without thermal data. In this study, we created a workflow to evaluate cloud and cloud shadow masking algorithms using cloud validation masks manually derived from both Landsat 7 Enhanced Thematic Mapper Plus (ETM+) and Landsat 8 OLI/TIRS data. We created a new validation dataset consisting of 96 Landsat 8 scenes, representing different biomes and proportions of cloud cover. We evaluated algorithm performance by overall accuracy, omission error, and commission error for both cloud and cloud shadow. We found that CFMask, C code based on the Function of Mask (Fmask) algorithm, and its confidence bands have the best overall accuracy among the many algorithms tested using our validation data. The Artificial Thermal-Automated Cloud Cover Algorithm (AT-ACCA) is the most accurate nonthermal-based algorithm. We give preference to CFMask for operational cloud and cloud shadow detection, as it is derived from a priori knowledge of physical phenomena and is operable without geographic restriction, making it useful for current and future land imaging missions without having to be retrained in a machine-learning environment.

## Dataset

This collection contains 96 Pre-Collection Landsat 8 Operational Land Imager (OLI) Thermal Infrared Sensor (TIRS) terrain-corrected (Level-1T) scenes, displayed in the biomes listed below. Manually generated cloud masks are used to validate cloud cover assessment algorithms, which in turn are intended to compute the percentage of cloud cover in each scene.

![image](https://github.com/user-attachments/assets/6ab655dc-aeda-4205-8aa8-dd43f9494728)


```yaml
name: hrc_whu
source: Landsat-8 OLI/TIRS
band: 9
resolution: 30m
pixel: ∼7000 × 6000
train: -
val: -
test: -
disk: 88GB
annotation: 
    - 0: Fill
    - 64: Cloud Shadow
    - 128: Clear
    - 192: Thin Cloud
    - 255: Cloud
scene: [Barren,Forest,Grass/Crops,Shrubland,Snow/Ice,Urban,Water,Wetlands]
```

## Citation

```bibtex
@article{l8biome,
  title = {Cloud detection algorithm comparison and validation for operational Landsat data products},
  journal = {Remote Sensing of Environment},
  volume = {194},
  pages = {379-390},
  year = {2017},
  issn = {0034-4257},
  doi = {https://doi.org/10.1016/j.rse.2017.03.026},
  url = {https://www.sciencedirect.com/science/article/pii/S0034425717301293},
  author = {Steve Foga and Pat L. Scaramuzza and Song Guo and Zhe Zhu and Ronald D. Dilley and Tim Beckmann and Gail L. Schmidt and John L. Dwyer and M. {Joseph Hughes} and Brady Laue},
  keywords = {Landsat, CFMask, Cloud detection, Cloud validation masks, Biome sampling, Data products},
}
```
