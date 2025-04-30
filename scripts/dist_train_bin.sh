#!/usr/bin/env bash

CONFIG=/root/caixin/RawSense/nolens_mmcls/configs/mobilefacenet/mobilefacenet_celeb_featureaffine01.py
GPUS=8
WORRDIR=logs/celeb_bin/
mkdir -p $WORRDIR
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29400}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py \
    $CONFIG \
    --seed 0 \
    --work-dir $WORRDIR \
    --launcher pytorch ${@:3}\
    
