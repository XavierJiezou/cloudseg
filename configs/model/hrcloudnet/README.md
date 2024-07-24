# High-Resolution Cloud Detection Network

> [High-Resolution Cloud Detection Network](https://arxiv.org/abs/2407.07365)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://arxiv.org/abs/2407.07365">Official Repo</a>

<a href="https://github.com/kunzhan/HR-cloud-Net">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

The complexity of clouds, particularly in terms of texture
detail at high resolutions, has not been well explored by most
existing cloud detection networks. This paper introduces
the High-Resolution Cloud Detection Network (HR-cloudNet), which utilizes a hierarchical high-resolution integration approach. HR-cloud-Net integrates a high-resolution
representation module, layer-wise cascaded feature fusion
module, and multi-resolution pyramid pooling module to
effectively capture complex cloud features. This architecture
preserves detailed cloud texture information while facilitating feature exchange across different resolutions, thereby
enhancing overall performance in cloud detection. Additionally, a novel approach is introduced wherein a student
view, trained on noisy augmented images, is supervised by a
teacher view processing normal images. This setup enables
the student to learn from cleaner supervisions provided by
the teacher, leading to improved performance. Extensive
evaluations on three optical satellite image cloud detection
datasets validate the superior performance of HR-cloud-Net
compared to existing methods.


<!-- [IMAGE] -->

<div align=center>
<img src="https://arxiv.org/html/2407.07365v1/x1.png" width="70%"/>
</div>

## Results and models

### CHLandSat-8 dataset


| method       | mae        | weight-F-measure | structure-measure |
|--------------|------------|------------------|-------------------|
| U-Net        | 0.1130     | 0.7448           | 0.7228            |
| PSPNet       | 0.0969     | 0.7989           | 0.7672            |
| SegNet       | 0.1023     | 0.7780           | 0.7540            |
| Cloud-Net    | 0.1012     | 0.7641           | 0.7368            |
| CDNet        | 0.1286     | 0.7222           | 0.7087            |
| CDNet-v2     | 0.1254     | 0.7350           | 0.7141            |
| HRNet        | **0.0737** | 0.8279           | **0.8141**        |
| GANet        | 0.0751     | **0.8396**       | 0.8106            |
| HR-cloud-Net | **0.0628** | **0.8503**       | **0.8337**        |

### 38-cloud dataset


| method       | mae        | weight-F-measure | structure-measure |
|--------------|------------|------------------|-------------------|
| U-Net        | 0.0638     | 0.7966           | 0.7845            |
| PSPNet       | 0.0653     | 0.7592           | 0.7766            |
| SegNet       | 0.0556     | 0.8002           | 0.8059            |
| Cloud-Net    | 0.0556     | 0.7615           | 0.7987            |
| CDNet        | 0.1057     | 0.7378           | 0.7270            |
| CDNet-v2     | 0.1084     | 0.7183           | 0.7213            |
| HRNet        | 0.0538     | 0.8086           | 0.8183            |
| GANet        | **0.0410** | **0.8159**       | **0.8342**        |
| HR-cloud-Net | **0.0395** | **0.8673**       | **0.8479**        |


### SPARCS dataset


| method       | mae        | weight-F-measure | structure-measure |
|--------------|------------|------------------|-------------------|
| U-Net        | 0.1314     | 0.3651           | 0.5416            |
| PSPNet       | 0.1263     | 0.3758           | 0.5414            |
| SegNet       | 0.1100     | 0.4697           | 0.5918            |
| Cloud-Net    | 0.1213     | 0.3804           | 0.5536            |
| CDNet        | 0.1157     | 0.4585           | 0.5919            |
| CDNet-v2     | 0.1219     | 0.4247           | 0.5704            |
| HRNet        | 0.1008     | 0.3742           | 0.5777            |
| GANet        | **0.0987** | **0.5134**       | **0.6210**        |
| HR-cloud-Net | **0.0833** | **0.5202**       | **0.6327**        |




## Citation

```bibtex
@InProceedings{LiJEI2024,
  author =    {Jingsheng Li and Tianxiang Xue and Jiayi Zhao and 
               Jingmin Ge and Yufang Min and Wei Su and Kun Zhan},
  title =     {High-Resolution Cloud Detection Network},
  booktitle = {Journal of Electronic Imaging},
  year =      {2024},
}
```