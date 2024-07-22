# Remote sensing image cloud detection using a shallow convolutional neural network

> [Remote sensing image cloud detection using a shallow convolutional neural network](https://www.sciencedirect.com/science/article/abs/pii/S0924271624000352?via%3Dihub#fn1)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/1921134176/Deeplearning-for-cloud-detection">Official Repo</a>

<a href="https://github.com/dfchai/SCNN">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

The state-of-the-art methods for cloud detection are dominated by deep convolutional neural networks (DCNNs). However, it is very expensive to train DCNNs for cloud detection and the trained DCNNs do not always perform well as expected. This paper proposes a shallow CNN (SCNN) by removing pooling/unpooling layers and normalization layers in DCNNs, retaining only three convolutional layers, and equipping them with 3 filters of 
1
×
1
,
1
×
1
,
3
×
3
 in spatial dimensions. It demonstrates that the three convolutional layers are sufficient for cloud detection. Since the label output by the SCNN for a pixel depends on a 3 × 3 patch around this pixel, the SCNN can be trained using some thousands 3 × 3 patches together with ground truth of their center pixels. It is very cheap to train a SCNN using some thousands 3 × 3 patches and to provide ground truth of their center pixels. Despite of its low cost, SCNN training is stabler than DCNN training, and the trained SCNN outperforms the representative state-of-the-art DCNNs for cloud detection. The same resolution of original image, feature maps and final label map assures that details are not lost as by pooling/unpooling in DCNNs. The border artifacts suffering from deep convolutional and pooling/unpooling layers are minimized by 3 convolutional layers with 
1
×
1
,
1
×
1
,
3
×
3
 filters. Incoherent patches suffering from patch-by-patch segmentation and batch normalization are eliminated by SCNN without normalization layers. Extensive experiments based on the L7 Irish, L8 Biome and GF1 WHU datasets are carried out to evaluate the proposed method and compare with state-of-the-art methods. The proposed SCNN promises to deal with images from any other sensors.






## Citation

```bibtex
@InProceedings{LiJEI2024,
  author =    {Dengfeng Chai and Jingfeng Huang and Minghui Wu and 
               Xiaoping Yang and Ruisheng Wang},
  title =     {Remote sensing image cloud detection using a shallow convolutional neural network},
  booktitle = {ISPRS Journal of Photogrammetry and Remote Sensing},
  year =      {2024},
}
```