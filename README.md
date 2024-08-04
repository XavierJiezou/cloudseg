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
