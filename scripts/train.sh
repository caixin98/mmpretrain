#!/usr/bin/env bash

CONFIG=$1

# WORRDIR is logs + the file name of CONFIG without .py
WORRDIR=$(echo logs/${CONFIG%.*} | sed 's/configs\///g')

mkdir -p $WORRDIR


python tools/train.py \
    $CONFIG \
    --seed 0 \
    --work-dir $WORRDIR \
    
