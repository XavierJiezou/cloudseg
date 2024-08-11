# Cloud Segmentation for Remote Sensing

[![demo](https://img.shields.io/badge/🤗HugginngFace-Spaces-orange)](https://huggingface.co/spaces/caixiaoshun/cloudseg)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/XavierJiezou/cloudseg#license)
[![contributors](https://img.shields.io/github/contributors/XavierJiezou/cloudseg.svg)](https://github.com/XavierJiezou/cloudseg/graphs/contributors)
[![Template](https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray)](https://github.com/ashleve/lightning-hydra-template)
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

## TODO

### 2024.08.11

- [ ] 评估指标计算框架由torchmetrics改为[mmeval](https://github.com/open-mmlab/mmeval/blob/main/mmeval/metrics/mean_iou.py)或[mmseg](https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/evaluation/metrics/iou_metric.py)
- [ ] 验证新的评估指标计算和原计算的结果是否基本一致
- [ ] 评估指标增加classwise支持

### 2024.08.10

- [x] 增加cloudsen12_high数据集支持

### 2024.08.09

- [x] 数据集下载

### 2024.08.08

- [x] 下载3个数据集

### 2024.08.07

- [x] 新增KappaMask模型，并完成在模型训练，wandb可视化，huggingface更新
- [x] 补充unetmobv2和kappamask文档

### 2024.08.06

- [x] 完成unetmobv2模型的训练，可视化，huggingface上传unetmobv2模型
- [x] 完成cloud-38数据集下载
 
## Datasets

```bash
cloudseg
├── src
├── configs
├── data
│   ├── hrcwhu
│   │   ├── train.txt
│   │   ├── test.txt
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── test
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── test
```

## Methods

- [UNet (MICCAI 2016)](configs/model/unet)
- [CDNetv1 (TGRS 2019)](configs/model/cdnetv1)
- [CDNetv2 (TGRS 2021)](configs/model/cdnetv2)
- [DBNet (TGRS 2022)](configs/model/dbnet)
- [HRCloudNet (JEI 2024)](configs/model/hrcloudnet)
- [MCDNet (JAG 2024)](configs/model/mcdnet)
- [SCNN (ISPRS 2024)](configs/model/scnn)

## Installation

```bash
git clone https://github.com/XavierJiezou/cloudseg.git
cd cloudseg
conda env create -f environment.yml
conda activate cloudseg
```

## Usage

**Train model with default configuration**

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

**Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)**

```bash
python src/train.py experiment=experiment_name.yaml
```

**Tranin Example**

```bash
python src/train.py experiment=hrcwhu_cdnetv1.yaml
```

**You can override any parameter from command line like this**

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

**Visualization in wandb**

```bash
python wand_vis.py --model-name model_name
```

**Example**

```bash
python wand_vis.py --model-name cdnetv1
```

## Docker

1. 编写 Docker 配置文件 [Dockerfile](Dockerfile)

```bash
# 使用一个基础的 Conda 镜像
FROM continuumio/miniconda3

# 将工作目录设置为 /app
WORKDIR /app

# 复制当前目录的内容到镜像的 /app 目录
COPY . /app

# 复制整个 Conda 环境到 Docker 镜像中
COPY ~/miniconda3/envs/cloudseg /opt/conda/envs/cloudseg

# 激活 Conda 环境并确保环境可用
RUN echo "source activate cloudseg" > ~/.bashrc
ENV PATH /opt/conda/envs/cloudseg/bin:$PATH

# 设置默认命令，进入bash并激活conda环境
CMD ["bash", "-c", "source activate cloudseg && exec bash"]
```

2. 构建 Docker 镜像

```bash
docker build -t xavierjiezou/cloudseg:latest .
```

3. 推送镜像

```bash
docker push xavierjiezou/cloudseg:latest
```

3. 运行 Docker 容器

```bash
docker pullxavierjiezou/cloudseg:latest
docker run -it xavierjiezou/cloudseg:latest
```
