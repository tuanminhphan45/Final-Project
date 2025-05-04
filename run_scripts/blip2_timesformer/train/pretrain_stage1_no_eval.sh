#!/bin/bash
export MASTER_PORT=55555  
export MASTER_ADDR=localhost
export CUDA_VISIBLE_DEVICES=0,1,2,3

export LAVIS_CACHE_ROOT="/storage/student10/vidcaption/LAVIS/cache"
cd /storage/student10/vidcaption/LAVIS
export PYTHONPATH=/storage/student10/vidcaption/LAVIS:$PYTHONPATH

# Cài đặt biến môi trường để tối ưu bộ nhớ CUDA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=${MASTER_PORT} \
    train.py \
    --cfg-path lavis/projects/blip2_timesformer/train/pretrain_stage1.yaml \
    --options run.evaluate=False  # Tắt đánh giá trong quá trình huấn luyện 