# Cloud Segmentation

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
- [Dual_Branch (TGRS 2022)](configs/model/dual_branch)
- [HrCloudNet (arxiv 2024)](configs/model/hrcloudnet)
- [McdNet (JAG 2024)](configs/model/mcdnet)
- [Scnn (ISPRS JOURNAL 2024)](configs/model/scnn)

## Installation



```bash
# clone project
git clone https://github.com/XavierJiezou/cloudseg.git
cd cloudseg

# [OPTIONAL] create conda environment
conda env create -f environment.yaml
conda activate cloudseg

# install pytorch according to instructions
# https://pytorch.org/get-started/
```

## Usage

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
