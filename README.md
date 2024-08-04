# Cloud Segmentation

[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/XavierJiezou/cloudseg#license)
[![contributors](https://img.shields.io/github/contributors/XavierJiezou/cloudseg.svg)](https://github.com/XavierJiezou/cloudseg/graphs/contributors)
[![Template](https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray)](https://github.com/ashleve/lightning-hydra-template)
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

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

## Supported Methods

- [UNet (MICCAI 2016)](configs/model/unet)
- [CDNetv1 (TGRS 2019)](configs/model/cdnetv1)
- [CDNetv2 (TGRS 2021)](configs/model/cdnetv2)
- [DBNet (TGRS 2022)](configs/model/dual_branch)
- [HrCloudNet (JEI 2024)](configs/model/hrcloudnet)
- [McdNet (International Journal of Applied Earth Observation and Geoinformation 2024)](configs/model/mcdnet)
- [Scnn (ISPRS 2024)](configs/model/scnn)

## Installation

```bash
git clone https://github.com/XavierJiezou/cloudseg.git
cd cloudseg
conda env create -f environment.yaml
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
