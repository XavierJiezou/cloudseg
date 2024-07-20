# Multimedia Deepfake Challenge

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/XavierJiezou/multimedia-deepfake-challenge
cd multimedia-deepfake-challenge

# [OPTIONAL] create conda environment
conda create -n deepfake python=3.9
conda activate deepfake

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/XavierJiezou/multimedia-deepfake-challenge
cd multimedia-deepfake-challenge

# create conda environment and install dependencies
conda env create -f environment.yaml -n deepfake

# activate conda environment
conda activate deepfake
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
