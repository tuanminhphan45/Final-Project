#!/bin/bash

# Distributed training trên 3 GPUs
python -m torch.distributed.run \
    --nproc_per_node=3 \
    train.py \
    --cfg-path lavis/projects/blip2_timesformer/train/pretrain_stage1.yaml
