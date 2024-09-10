#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py experiment=hrc_whu/scnn.yaml trainer.devices=[3] trainer.max_epochs=1
python src/train.py experiment=cloudsen12_high_l1c/scnn.yaml trainer.devices=[3] trainer.max_epochs=1
python src/train.py experiment=cloudsen12_high_l2a/scnn.yaml trainer.devices=[3] trainer.max_epochs=1
python src/train.py experiment=l8_biome/scnn.yaml trainer.devices=[3] trainer.max_epochs=1
python src/train.py experiment=gf12ms_whu_gf1/scnn.yaml trainer.devices=[3] trainer.max_epochs=1
python src/train.py experiment=gf12ms_whu_gf2/scnn.yaml trainer.devices=[3] trainer.max_epochs=1

python src/train.py experiment=hrc_whu/cdnetv1.yaml trainer.devices=[3] trainer.max_epochs=1
python src/train.py experiment=hrc_whu/cdnetv2.yaml trainer.devices=[3] trainer.max_epochs=1
python src/train.py experiment=hrc_whu/mcdnet.yaml trainer.devices=[3] trainer.max_epochs=1
python src/train.py experiment=hrc_whu/dbnet.yaml trainer.devices=[3] trainer.max_epochs=1
python src/train.py experiment=hrc_whu/hrcloudnet.yaml trainer.devices=[3] trainer.max_epochs=1
python src/train.py experiment=hrc_whu/kappamask.yaml trainer.devices=[3] trainer.max_epochs=1
python src/train.py experiment=hrc_whu/unetmobv2.yaml trainer.devices=[3] trainer.max_epochs=1
