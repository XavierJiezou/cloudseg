# KappaMask: AI-Based Cloudmask Processor for Sentinel-2

> [KappaMask: AI-Based Cloudmask Processor for Sentinel-2](https://www.mdpi.com/2072-4292/13/20/4100)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/cloudsen12/models/tree/9c50b5da275df122165039ef788de830c1ea73a2/kappamask">Official Repo</a>

## Abstract

<!-- [ABSTRACT] -->

The Copernicus Sentinel-2 mission operated by the European Space Agency (ESA) provides comprehensive and continuous multi-spectral observations of all the Earth’s land surface since mid-2015. Clouds and cloud shadows significantly decrease the usability of optical satellite data, especially in agricultural applications; therefore, an accurate and reliable cloud mask is mandatory for effective EO optical data exploitation. During the last few years, image segmentation techniques have developed rapidly with the exploitation of neural network capabilities. With this perspective, the KappaMask processor using U-Net architecture was developed with the ability to generate a classification mask over northern latitudes into the following classes: clear, cloud shadow, semi-transparent cloud (thin clouds), cloud and invalid. For training, a Sentinel-2 dataset covering the Northern European terrestrial area was labelled. KappaMask provides a 10 m classification mask for Sentinel-2 Level-2A (L2A) and Level-1C (L1C) products. The total dice coefficient on the test dataset, which was not seen by the model at any stage, was 80% for KappaMask L2A and 76% for KappaMask L1C for clear, cloud shadow, semi-transparent and cloud classes. A comparison with rule-based cloud mask methods was then performed on the same test dataset, where Sen2Cor reached 59% dice coefficient for clear, cloud shadow, semi-transparent and cloud classes, Fmask reached 61% for clear, cloud shadow and cloud classes and Maja reached 51% for clear and cloud classes. The closest machine learning open-source cloud classification mask, S2cloudless, had a 63% dice coefficient providing only cloud and clear classes, while KappaMask L2A, with a more complex classification schema, outperformed S2cloudless by 17%.
<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/user-attachments/assets/9c50377a-519d-48e3-8302-59b4f39ffa6f" width="70%"/>
</div>


## Results and models

### Output correspondence for different cloud classification masks.

|           Sen2Cor          |           CMIX           |         KappaMask        |      Fmask      | S2Cloudless |       FMSC      | **UAlow%** | **UAmiddle%** | **UAhigh%** |
|:--------------------------:|:------------------------:|:------------------------:|:---------------:|:-----------:|:---------------:|------------|---------------|-------------|
|          0 No data         |                          |         0 Missing        |                 |             |                 | 0.13       | 4.39          | 95.48       |
|  1 Saturated or defective  |                          |         0 Missing        |                 |             |                 | 0.26       | 25.03         | 74.71       |
|     2 Dark area pixels     |                          |          1 Clear         |                 |             |                 | 1.56       | 63.04         | 35.41       |
|       3 Cloud shadows      |      4 Cloud shadows     |      2 Cloud shadows     | 2 Cloud shadows |             | 2 Cloud shadows | 0.65       | 38.38         | 60.97       |
|        4 Vegetation        |          1 Clear         |          1 Clear         |     0 Clear     |   0 Clear   |     0 Clear     | 0.26       | 52.65         | 47.09       |
|       5 Not vegetated      |                          |          1 Clear         |                 |             |                 | 0.65       | 31.5          | 67.84       |
|           6 Water          |                          |          1 Clear         |     1 Water     |             |                 | 1.58       | 20.05         | 78.36       |
|       7 Unclassified       |                          |        5 Undefined       |                 |             |                 | 1.39       | 37.62         | 60.99       |
| 8 Cloud medium probability |                          |          4 Cloud         |                 |             |                 | 1.62       | 12.58         | 85.79       |
|  9 Cloud high probability  |          2 Cloud         |          4 Cloud         |     4 Cloud     |   1 Cloud   |     1 Cloud     | 0.82       | 9.43          | 89.75       |
|       10 Thin cirrus       | 3 Semi-transparent cloud | 3 Semi-transparent cloud |                 |             |                 | 0.60       | 9.97          | 89.43       |
|           11 Snow          |                          |          1 Clear         |      3 Snow     | 67.16       | 23.96           | 7.99       | 46.65         | 45.36       |
|                            | KappaMask L2A            | 0.64                     | 37.28           | 59.76       | 2.96            | 12.24      | 36.9          | 50.85       |
|                            | KappaMask L1C            | 0.74                     | 30.03           | 60.95       | 9.02            | 20.67      | 59.36         | 19.97       |
|                            | Fmask                    | 0.72                     | 22.34           | 76.04       | 1.63            | 14.53      | 77.06         | 8.41        |
|                            | Sen2Cor                  | 0.51                     | 64.5            | 35.21       | 0.30            | 6.90       | 18.10         | 75.00       |
| **c. valid/invalid**       | Human level              | 0.99                     | 1.03            | 14.8        | 84.17           | 0.13       | 2.33          | 97.55       |
|                            | UNetMobV2                | 0.91                     | 0.77            | 28.57       | 70.66           | 0.00       | 17.14         | 82.86       |
|                            | KappaMask L2A            | 0.75                     | 2.96            | 39.38       | 57.66           | 1.29       | 44.32         | 54.39       |
|                            | KappaMask L1C            | 0.81                     | 3.99            | 47.62       | 48.39           | 0.65       | 32.64         | 66.71       |
|                            | Fmask                    | 0.81                     | 4.89            | 45.43       | 49.68           | 0.26       | 44.34         | 55.39       |
|                            | Sen2Cor                  | 0.67                     | 13.77           | 69.63       | 16.6            | 1.05       | 18.58         | 80.37       |

###  Dice coefficient evaluation performed on the test dataset for KappaMask Level-2A, KappaMask Level-1C, Sen2Cor, Fmask and MAJA cloud classification maps


| Dice Coefficient | KappaMask L2A | KappaMask L1C | Sen2Cor | Fmask | MAJA |
|:----------------:|:-------------:|:-------------:|:-------:|:-----:|:----:|
|       Clear      |      82%      |      75%      |   72%   |  75%  |  56% |
|   Cloud shadow   |      72%      |      69%      |   52%   |  49%  |   -  |
| Semi-transparent |      78%      |      75%      |   49%   |   -   |   -  |
|       Cloud      |      86%      |      84%      |   62%   |  60%  |  46% |
|    All classes   |      80%      |      76%      |   59%   |  61%  |  51% |


### Precision evaluation performed on the test dataset for KappaMask Level-2A, KappaMask Level-1C, Sen2Cor, Fmask and MAJA cloud classification maps


|     Precision    | KappaMask L2A | KappaMask L1C | Sen2Cor | Fmask | MAJA |
|:----------------:|:-------------:|:-------------:|:-------:|:-----:|:----:|
|       Clear      |      75%      |      79%      |   60%   |  66%  |  64% |
|   Cloud shadow   |      82%      |      79%      |   87%   |  51%  |   -  |
| Semi-transparent |      83%      |      71%      |   78%   |   -   |   -  |
|       Cloud      |      85%      |      83%      |   57%   |  44%  |  35% |
|    All classes   |      81%      |      78%      |   71%   |  54%  |  50% |


### Recall evaluation performed on the test dataset for KappaMask Level-2A, KappaMask Level-1C, Sen2Cor, Fmask and MAJA cloud classification maps.

|      Recall      | KappaMask L2A | KappaMask L1C | Sen2Cor | Fmask | MAJA |
|:----------------:|:-------------:|:-------------:|:-------:|:-----:|:----:|
|       Clear      |      91%      |      71%      |   90%   |  86%  |  50% |
|   Cloud shadow   |      64%      |      61%      |   37%   |  48%  |   -  |
| Semi-transparent |      74%      |      80%      |   36%   |   -   |   -  |
|       Cloud      |      87%      |      85%      |   67%   |  60%  |  65% |
|    All classes   |      79%      |      74%      |   58%   |  65%  |  58% |


### Overall accuracy evaluation performed on the test dataset for KappaMask Level-2A, KappaMask Level-1C, Sen2Cor, Fmask and MAJA cloud classification maps.

| Overall Accuracy | KappaMask L2A | KappaMask L1C | Sen2Cor | Fmask | MAJA |
|:----------------:|:-------------:|:-------------:|:-------:|:-----:|:----:|
|       Clear      |      89%      |      86%      |   81%   |  84%  |  79% |
|   Cloud shadow   |      96%      |      95%      |   95%   |  92%  |   -  |
| Semi-transparent |      85%      |      79%      |   72%   |   -   |   -  |
|       Cloud      |      92%      |      91%      |   78%   |  67%  |  63% |
|    All classes   |      91%      |      88%      |   82%   |  81%  |  71% |


### Dice coefficient evaluation performed on the test dataset for KappaMask Level-2A, KappaMask Level-1C, S2cloudless and DL_L8S2_UV cloud classification maps.

| Dice Coefficient | KappaMask L2A | KappaMask L1C | S2cloudless | DL_L8S2_UV |
|:----------------:|:-------------:|:-------------:|:-----------:|:----------:|
|       Clear      |      82%      |      75%      |     69%     |     56%    |
|       Cloud      |      86%      |      84%      |     57%     |     67%    |
|    All classes   |      84%      |      80%      |     63%     |     62%    |


### Training experiments for different model architectures for the L2A model.

| Architecture | U-Net Level of Depth | Number of Input Filters | Max Dice Coefficient on Validation Set |
|:------------:|:--------------------:|:-----------------------:|:--------------------------------------:|
|     U-Net    |           5          |            32           |                  83.9%                 |
|              |                      |            64           |                  84.0%                 |
|              |                      |           128           |                  84.1%                 |
|              |           6          |            32           |                  80.7%                 |
|              |                      |            64           |                  80.8%                 |
|              |                      |           128           |                  82.9%                 |
|              |           7          |            32           |                  75.1%                 |
|              |                      |            64           |                  83.1%                 |
|    U-Net++   |           5          |            64           |                  75.9%                 |

### Time comparison performed on one whole Sentinel-2 Level-1C product inference.

|              | KappaMask on GPU | KappaMask on CPU | Fmask | S2cloudless | DL_L8S2_UV |
|:------------:|:----------------:|:----------------:|:-----:|:-----------:|:----------:|
| Running time |       03:57      |       10:08      | 06:32 |    17:34    |    03:33   |

## Citation

```bibtex
@article{kappamask,
title={KappaMask: AI-based cloudmask processor for Sentinel-2},
author={Domnich, M. and Sünter, I. and Trofimov, H. and others},
journal={Remote Sensing},
volume={13},
number={20},
pages={4100},
year={2021},
publisher={MDPI}
}
```
