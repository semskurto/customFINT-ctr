#!/bin/sh
# Author: zhao zhishan
# Created Time : 2021-01-14 11:26
# File Name: run.sh
# Description:

model_name=mlp
data=avazu_tmp
#data=criteo
#data=kdd2012
version=mlp3
gpu=0
field_num=22
#field_num=21
#field_num=13

python train.py --feature_size_file ./data/${data}/data.feat_info \
    --train_data ./data/${data}/train_data.tfrecord \
    --val_data ./data/${data}/val_data.tfrecord \
    --eval_data ./data/${data}/eval_data.tfrecord \
    --test_data ./data/${data}/test_data.tfrecord \
    --model_name $model_name \
    --model_dir ./checkpoints/${version}/${data}/ \
    --learning_rate 0.01 \
    --epoch 150 \
    --l2_reg 0.0 \
    --dropout_rate 0.3 \
    --batch_size 256 \
    --emb_size 16 \
    --eval_step 5000 \
    --field_num ${field_num} \
    --version ${version} \
    --gpu ${gpu} \
