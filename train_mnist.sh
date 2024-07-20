#!/bin/bash
export http_proxy="http://hk-mmhttpproxy.woa.com:11113"
export https_proxy="$http_proxy"
export WANDB_API_KEY="76211aa17d75da9ddab7b8cba5743454194fe1d5"

set -xe # 得脚本在执行时会输出每个命令，并且在任何命令失败时立即终止执行

cd $(dirname $0) # 让当前脚本在执行时切换到其所在的目录，以确保后续的操作在正确的工作目录下执行

umask 0 # 新创建的文件将具有最大的权限（读、写、执行权限）

export LANG=en_US.utf8
export HOME=$(pwd)
# export HF_HUB_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export TORCH_HOME=/apdcephfs_cq2/share_47076/yshzhu/models
# export XDG_CACHE_HOME=/apdcephfs_cq2/share_47076/yshzhu/models
# export TRANSFORMERS_CACHE=/apdcephfs_cq2/share_47076/yshzhu/models/huggingface/hub

# export WANDB_BASE_URL='http://9.141.138.229'
# export WANDB_API_KEY='local-b6abe87ae83b8cfd8eb6aca4e659b414d9135259'
# export WANDB_MODE=offline

# echo "NODE_RANK: $RANK"
# echo "GPU_NUM: $GPU_NUM"
# echo "WORLD_SIZE: $WORLD_SIZE"
# echo "MASTER_ADDR: $MASTER_ADDR"
# echo "MASTER_PORT: $MASTER_PORT"
# echo "CUDA_HOME: $CUDA_HOME"
# export CUDA_VISIBLE_DEVICES="0,1" CUDA_LAUNCH_BLOCKING=1

EXP_NAME=mnist-simple_dense_net
echo "begin processing: $EXP_NAME ..."
PYTHON_EXE="/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/zouxuechao/miniconda3/envs/myenv/bin/python"

$PYTHON_EXE src/train.py experiment=cnn
#  --name=$EXP_NAME \
#  --pretrained_model_name_or_path=$MODEL_NAME \
#  --instance_data_dir=$INSTANCE_DIR \
#  --output_dir=$OUTPUT_DIR \
#  --prompt_suffix=",卡通,白色透明背景,专辑,微信官方,精致原创" \
#  --instance_prompt="sks" \
#  --validation_prompt="sks嘻嘻嘻表情 || sks几点表情 || sks心疼表情 \
#  --resolution=256 \
#  --train_batch_size=4 \
#  --lr_scheduler="constant" \
#  --lr_warmup_steps=0 \
#  --use_lora \
#  --lora_r 16 \
#  --lora_alpha 27 \
#  --lora_text_encoder_r 16 \
#  --lora_text_encoder_alpha 17 \
#  --learning_rate=1e-4 \
#  --gradient_accumulation_steps=1 \
#  --max_train_steps=2001 \
#  --checkpointing_steps=200 \
#  --validation_steps=200 \
#  --report_to=wandb

