# Cloud Segmentation for Remote Sensing

[![demo](https://img.shields.io/badge/🤗HugginngFace-Spaces-orange)](https://huggingface.co/spaces/caixiaoshun/cloudseg)
[![Template](https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray)](https://github.com/ashleve/lightning-hydra-template)
<!--[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://arxiv.org)-->
<!--[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)]([https://papers.nips.cc/paper/2020](https://arxiv.org))-->

**CloudSeg is a repository containing the implementation of methods compared in the paper *"Adapting Vision Foundation Models for Robust Cloud Segmentation in Remote Sensing Images."* It includes implementations of CDNetv1 (TGRS 2019), CDNetv2 (TGRS 2021), DBNet (TGRS 2022), HRCloudNet (JEI 2024), MCDNet (JAG 2024), SCNN (ISPRS 2024),KappaMask (RS 2021) and UNetMobv2 (Scientific Data 2022). We have open-sourced the pretrained weights for these methods on various datasets, available at [Hugging Face](https://huggingface.co/XavierJiezou/cloud-adapter-models).**

## Leaderboard (mIoU, %, ↑)

|   **Methods**  | **HRC_WHU** | **GF12MS_WHU_GF1** | **GF12MS_WHU_GF2** | **CloudSEN12_High_L1C** | **CloudSEN12_High_L2A** | **L8_Biome** |
|:--------------:|:-----------:|:------------------:|:------------------:|:-----------------------:|:-----------------------:|:------------:|
|    **SCNN**    |    57.22    |        81.68       |        76.99       |          22.75          |          28.76          |     32.38    |
|   **CDNetv1**  |    77.79    |        81.82       |        78.20       |          60.35          |          62.39          |     34.58    |
|   **CDNetv2**  |    76.75    |        84.93       |        78.84       |          65.60          |          66.05          |     43.63    |
|   **MCDNet**   |    53.50    |        85.16       |        78.36       |          44.80          |          46.52          |     33.85    |
|  **UNetMobv2** |    79.91    |        91.71       |      **80.44**     |        **71.65**        |        **70.36**        |     47.76    |
|    **DBNet**   |    77.78    |        91.36       |        78.68       |          65.52          |          65.65          |   **51.41**  |
| **HRCloudNet** |  **83.44**  |        91.86       |        75.57       |          68.26          |          68.35          |     43.51    |
|  **KappaMask** |    67.48    |      **92.42**     |        72.00       |          41.27          |          45.28          |     42.12    |

## Installation

```bash
git clone https://github.com/XavierJiezou/cloudseg.git
cd cloudseg
conda create -n cloudseg python=3.11.7
conda activate cloudseg
pip install -r requirements.txt
```

**OR**  

We have uploaded the conda virtual environment used in our experiments to [Hugging Face](https://huggingface.co/XavierJiezou/cloudseg-models/blob/main/envs.tar.gz). You can download it directly from the link, extract the files, and activate the environment using the following commands:  

```bash
mkdir envs
tar -zxvf envs.tar.gz -C envs
source env/bin/activate
````

## Datasets

You can download all datasets from [Hugging Face: CloudSeg Datasets](https://huggingface.co/datasets/XavierJiezou/cloudseg-datasets). The available datasets include:  
- **[L8_Biome (RSE 2017)](configs/data/l8_biome)**  
- **[HRC_WHU (ISPRS 2019)](configs/data/hrc_whu)**  
- **[CloudSEN12_High (Scientific Data 2022)](configs/data/cloudsen12_high)**  
- **[GF12MS_WHU (TGRS 2024)](configs/data/gf12ms_whu)**  

### Directory Structure

Below is an overview of the directory structure:  

```bash
cloudseg
├── src
├── configs
├── ...
├── data
│   ├── cloudsen12_high
│   ├── l8_biome
│   ├── gf12ms_whu
│   ├── hrc_whu
```

<details>
<summary>CloudSEN12_High</summary>

```
triain
├── EXTRA_*.dat
├── L1C_B*.dat
├── L2A_*.dat
├── LABEL_*.data
├── S1_*.data
├── metadata.csv
val
├── EXTRA_*.dat
├── L1C_B*.dat
├── L2A_*.dat
├── LABEL_*.data
├── S1_*.data
├── metadata.csv
test
├── EXTRA_*.dat
├── L1C_B*.dat
├── L2A_*.dat
├── LABEL_*.data
├── S1_*.data
├── metadata.csv
```
</details>

<details>
<summary>L8_Biome</summary>

```
train.txt
val.txt
test.txt
img_dir
├── train
├── val
├── test
ann_dir
├── train
├── val
├── test
```
</details>

<details>
<summary>GF12MS_WHU</summary>

```
GF1MS-WHU
├── TestBlock250
│   ├── *_Mask.tif
│   ├── *.tiff
├── TrainBlock250
│   ├── *_Mask.tif
│   ├── *.tiff
├── TestList.txt
├── TrainList.txt
GF2MS-WHU
├── TestBlock250
│   ├── *_Mask.tif
│   ├── *.tiff
├── TrainBlock250
│   ├── *_Mask.tif
│   ├── *.tiff
├── TestList.txt
├── TrainList.txt
```
</details>

<details>
<summary>HRC_WHU</summary>

```
train.txt
test.txt
img_dir
├── train
├── test
ann_dir
├── train
├── test
```
</details>

## Usage

**Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)**

```bash
python src/train.py experiment=hrc_whu_cdnetv1.yaml
```

**You can override any parameter from command line like this**

```bash
python src/train.py trainer.devices=["1"]
```
- In this example, the `trainer.devices` parameter is overridden to use GPU 1 for training.

### Model Evaluation and Visualization Guide

Below are instructions for running evaluation and visualization scripts for various models on datasets:

---

#### 1. General Evaluation Command
To evaluate the performance of models on a specified dataset:
```bash
python src/eval/eval_on_experiment.py --experiment_name=dataset_name --gpu="cuda:0"
```
- `--experiment_name=dataset_name`: Specifies the name of the dataset.
- `--gpu="cuda:0"`: Specifies the device to use for running the evaluation.

---

#### 2. Scene-Based Evaluation (for L8_Biome Dataset)
To evaluate the model's performance on **L8_Biome** dataset by scenes:
```bash
python src/eval/eval_l8_scene.py --gpu="cuda:0" --root="dataset_path"
```
- `--gpu="cuda:0"`: Specifies the device to use for running the evaluation.
- `--root="dataset_path"`: Specifies the dataset path.

## Methods

- [CDNetv1 (TGRS 2019)](configs/model/cdnetv1)
- [KappaMask (RS 2021)](configs/model/kappamask)
- [CDNetv2 (TGRS 2021)](configs/model/cdnetv2)
- [DBNet (TGRS 2022)](configs/model/dbnet)
- [UNetMobv2 (Scientific Data 2022)](configs/model/unetmobv2)
- [SCNN (ISPRS 2024)](configs/model/scnn)
- [MCDNet (JAG 2024)](configs/model/mcdnet)
- [HRCloudNet (JEI 2024)](configs/model/hrcloudnet)

## Gradio Demo

We provide a **Gradio Demo Application** for testing the methods in this repository. You can choose to run the demo locally or access it directly through our Hugging Face Space.

### Option 1: Run Locally

1. Clone this repository:
   ```bash
   git clone https://huggingface.co/spaces/caixiaoshun/cloudseg
   cd cloudseg
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
3. Start the Gradio demo:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to `http://127.0.0.1:7860` to interact with the demo.

### Option 2: Access on Hugging Face Space

You can also try the demo online without any setup:
[https://huggingface.co/spaces/caixiaoshun/cloudseg](https://huggingface.co/spaces/caixiaoshun/cloudseg)

## Citation

If you use our code or models in your research, please cite with:

```latex
@article{cloud-adapter,
  title={Adapting Vision Foundation Models for Robust Cloud Segmentation in Remote Sensing Images},
  author={Xuechao Zou and Shun Zhang and Kai Li and Shiying Wang and Junliang Xing and Lei Jin and Congyan Lang and Pin Tao},
  year={2024},
  eprint={2411.13127},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2411.13127}
}
```


