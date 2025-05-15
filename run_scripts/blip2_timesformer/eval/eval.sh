#!/bin/bash
export MASTER_PORT=55555
export MASTER_ADDR=localhost
export CUDA_VISIBLE_DEVICES=1,2,3

export LAVIS_CACHE_ROOT="/storage/student10/vidcaption/LAVIS/cache"
cd /storage/student10/vidcaption/LAVIS
export PYTHONPATH=/storage/student10/vidcaption/LAVIS:$PYTHONPATH

python -m torch.distributed.run --nproc_per_node=8 evaluate.py --cfg-path lavis/projects/blip2_timesformer/eval/eval_msrvtt.yaml
