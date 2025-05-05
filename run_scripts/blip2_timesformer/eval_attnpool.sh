#!/bin/bash
export MASTER_PORT=25557  # Sử dụng port khác để tránh xung đột
export MASTER_ADDR=localhost
export CUDA_VISIBLE_DEVICES=0  # Chỉ sử dụng một GPU cho đánh giá

export LAVIS_CACHE_ROOT="/storage/student10/vidcaption/LAVIS/cache"
cd /storage/student10/vidcaption/LAVIS
export PYTHONPATH=/storage/student10/vidcaption/LAVIS:$PYTHONPATH

# Cài đặt biến môi trường để tối ưu bộ nhớ CUDA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Bắt đầu đánh giá mô hình BLIP2TimeSformer với Temporal Attention Pooling..."

# Đánh giá mô hình
python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port=${MASTER_PORT} \
    evaluate.py \
    --cfg-path lavis/projects/blip2_timesformer/train/attnpool.yaml \
    --options run.evaluate=True run.batch_size_eval=4

# Hiển thị một số ví dụ về attention weights
echo "Hiển thị attention weights visualization cho một số mẫu video..."

python <<EOF
import torch
import matplotlib.pyplot as plt
import numpy as np
from lavis.models import load_model_and_preprocess
from lavis.common.registry import registry
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)

# Load mô hình và bộ xử lý
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_timesformer_attnpool", 
    model_type="pretrain", 
    is_eval=True, 
    device=device
)

# Tạo một hàm để trích xuất và hiển thị attention weights
def extract_attn_weights(model, video_path):
    # Xử lý video
    video = vis_processors["eval"](video_path)
    video = video.unsqueeze(0).to(device)
    
    # Trích xuất attention weights - đây là để minh họa, cần sửa đổi mô hình để lưu weights
    with torch.no_grad():
        video_embeds_raw = model.visual_encoder.forward_features(video)
        B = video_embeds_raw.shape[0]
        D = video_embeds_raw.shape[-1]
        
        cls_token = video_embeds_raw[:, 0:1, :]
        patch_tokens = video_embeds_raw[:, 1:, :]
        
        H = W = model.img_size // 16
        N = H * W
        T = model.num_frames
        
        patch_tokens = patch_tokens.reshape(B, T, N, D)
        
        # Lấy frame representations
        frame_repr = patch_tokens.mean(dim=2)
        scores = model.temporal_pool.attn_fc(frame_repr).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        
        return weights[0].cpu().numpy()

# Thử với một số video từ tập test
try:
    import os
    test_video_dir = "/storage/student10/vidcaption/LAVIS/datasets/msrvtt/videos"
    sample_videos = sorted(os.listdir(test_video_dir))[:5]  # Lấy 5 video đầu tiên
    
    plt.figure(figsize=(15, 10))
    for i, video_name in enumerate(sample_videos):
        video_path = os.path.join(test_video_dir, video_name)
        try:
            weights = extract_attn_weights(model, video_path)
            
            plt.subplot(len(sample_videos), 1, i+1)
            plt.bar(range(len(weights)), weights)
            plt.title(f"Video: {video_name}")
            plt.xlabel("Frame Index")
            plt.ylabel("Attention Weight")
            
            # Generate caption
            caption = model.generate({"video": vis_processors["eval"](video_path).unsqueeze(0).to(device)})[0]
            plt.figtext(0.5, 0.01 + i*0.2, f"Caption: {caption}", wrap=True, horizontalalignment='center', fontsize=10)
            
            print(f"Video: {video_name}")
            print(f"Attention weights: {weights}")
            print(f"Caption: {caption}")
            print("-" * 50)
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
    
    plt.tight_layout()
    plt.savefig("temporal_attention_visualization.png")
    print("Đã lưu visualization tại temporal_attention_visualization.png")
except Exception as e:
    print(f"Error during visualization: {e}")
EOF 