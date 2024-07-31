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
- [CDNetv1](configs/model/cdnetv1)
- [Dual_Branch](configs/model/dual_branch)
- [HrCloudNet](configs/model/hrcloudnet)
- [McdNet](configs/model/mcdnet)
- [Scnn](configs/model/scnn)

## Installation

### Pip

```bash
# clone project
git clone https://github.com/XavierJiezou/cloudseg.git
cd cloudseg

# [OPTIONAL] create pip environment
pip -m venv venv
source venv/bin/activate

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

### Conda

```bash
# clone project
git clone https://github.com/XavierJiezou/cloudseg.git
cd cloudseg

# [OPTIONAL] create conda environment
conda env create -f environment.yaml -n cloudseg
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
