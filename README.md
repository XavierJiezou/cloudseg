# Cloud Segmentation for Remote Sensing

[![demo](https://img.shields.io/badge/🤗HugginngFace-Spaces-orange)](https://huggingface.co/spaces/caixiaoshun/cloudseg)
[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/XavierJiezou/cloudseg#license)
[![contributors](https://img.shields.io/github/contributors/XavierJiezou/cloudseg.svg)](https://github.com/XavierJiezou/cloudseg/graphs/contributors)
[![Template](https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray)](https://github.com/ashleve/lightning-hydra-template)
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

## TODO

### 2024.08.08

### 2024.08.07

- [x] 新增KappaMask模型，并完成在模型训练，wandb可视化，huggingface更新
- [ ] 补充unetmobv2和kappamask文档

### 2024.08.06

- [ ] 增加另外三个数据集的支持
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

# 复制环境文件到镜像中
COPY environment.yml /tmp/environment.yml

# 创建 Conda 环境
RUN conda env create -f /tmp/environment.yml

# 激活 Conda 环境并确保环境可用
RUN echo "source activate cloudseg" > ~/.bashrc
ENV PATH /opt/conda/envs/cloudseg/bin:$PATH

# 将工作目录设置为 /app
WORKDIR /app

# 复制当前目录的内容到镜像的 /app 目录
COPY . /app

# 设置默认命令
CMD ["python", "-c 'import torch; print(torch.cuda.is_available())'"]
```

2. 构建 Docker 镜像

```bash
docker build -t cloudseg .
```

3. 运行 Docker 容器

```bash
docker run -it cloudseg
```