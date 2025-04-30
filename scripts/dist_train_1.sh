#!/usr/bin/env bash

CONFIG=$1
GPUS=8
# WORRDIR is logs + the file name of CONFIG without .py
WORRDIR=logs/$(basename "${CONFIG%.*}")
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
    
