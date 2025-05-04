#!/bin/bash
export MASTER_PORT=25556  # Sử dụng port khác để tránh xung đột
export MASTER_ADDR=localhost
export CUDA_VISIBLE_DEVICES=0  # Chỉ sử dụng một GPU cho đánh giá

export LAVIS_CACHE_ROOT="/storage/student10/vidcaption/LAVIS/cache"
cd /storage/student10/vidcaption/LAVIS
export PYTHONPATH=/storage/student10/vidcaption/LAVIS:$PYTHONPATH

# Cài đặt biến môi trường để tối ưu bộ nhớ CUDA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Tạo mẫu đánh giá
# Tạo một tập nhỏ để test mô hình trước
python <<EOF
import json
import os

# Đường dẫn tới file annotation gốc
input_file = "/storage/student10/vidcaption/LAVIS/cache/msrvtt_caption_gt/msrvtt_caption_test_annotations.json"
output_file = "/storage/student10/vidcaption/LAVIS/cache/msrvtt_caption_gt/msrvtt_caption_test_mini.json"

# Đảm bảo thư mục tồn tại
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Đọc và lưu subset
with open(input_file, 'r') as f:
    data = json.load(f)

# Lấy 10 mẫu đầu tiên
mini_data = {
    "images": data["images"][:10],
    "annotations": [ann for ann in data["annotations"] if ann["image_id"] in [img["id"] for img in data["images"][:10]]]
}

# Lưu file
with open(output_file, 'w') as f:
    json.dump(mini_data, f)

print(f"Đã tạo tập mini-test với {len(mini_data['images'])} hình ảnh và {len(mini_data['annotations'])} annotation")
EOF

# Chạy đánh giá với mô hình tối ưu bộ nhớ
python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port=${MASTER_PORT} \
    evaluate.py \
    --cfg-path lavis/projects/blip2_timesformer/train/pretrain_stage1_eval.yaml \
    --options model.arch=blip2_timesformer_memory_optimized run.batch_size_eval=1 run.num_beams=1 \
    run.annotation_file=/storage/student10/vidcaption/LAVIS/cache/msrvtt_caption_gt/msrvtt_caption_test_mini.json 