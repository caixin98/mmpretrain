#!/usr/bin/env bash

CONFIG=/root/caixin/RawSense/nolens_mmcls/configs/a_optical_face/optical_mobilefacenet_celeb_false.py
GPUS=8
WORRDIR=logs/optical_mobilefacenet_celeb_required_grad=false/
mkdir -p $WORRDIR
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
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
    
