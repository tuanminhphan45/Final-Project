#!/bin/bash
export MASTER_PORT=25555  # Sử dụng port khác để tránh xung đột
export MASTER_ADDR=localhost
export CUDA_VISIBLE_DEVICES=0  # Chỉ sử dụng một GPU cho đánh giá

export LAVIS_CACHE_ROOT="/storage/student10/vidcaption/LAVIS/cache"
cd /storage/student10/vidcaption/LAVIS
export PYTHONPATH=/storage/student10/vidcaption/LAVIS:$PYTHONPATH

# Cài đặt biến môi trường để tối ưu bộ nhớ CUDA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Chạy đánh giá với cấu hình tối ưu hóa bộ nhớ
python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port=${MASTER_PORT} \
    evaluate.py \
    --cfg-path lavis/projects/blip2_timesformer/train/pretrain_stage1_eval.yaml \
    --options model.num_query_token=32 run.batch_size_eval=2 run.num_beams=2 run.max_len=20 