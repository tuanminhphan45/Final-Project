"""
Copyright (c) 2023, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import torch
import logging
import decord
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from lavis.models import load_model_and_preprocess
from lavis.common.registry import registry
from lavis.processors.video_processors import PrepareTimeSformerVideo

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("blip2_timesformer_inference")

def parse_args():
    parser = argparse.ArgumentParser(description="BLIP2 TimeSformer Inference")
    parser.add_argument("--video", required=True, help="Đường dẫn đến file video")
    parser.add_argument("--model_type", default="pretrain", help="Loại model (pretrain, caption, ...)")
    parser.add_argument("--checkpoint", help="Đường dẫn đến file checkpoint model")
    parser.add_argument("--num_beams", type=int, default=5, help="Số beam cho beam search")
    parser.add_argument("--max_length", type=int, default=30, help="Độ dài tối đa của caption")
    parser.add_argument("--min_length", type=int, default=8, help="Độ dài tối thiểu của caption")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Hệ số phạt lặp lại")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Hệ số phạt độ dài")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p cho nucleus sampling")
    parser.add_argument("--temperature", type=float, default=0.7, help="Nhiệt độ cho sampling")
    parser.add_argument("--use_nucleus_sampling", action="store_true", help="Sử dụng nucleus sampling thay vì beam search")
    parser.add_argument("--n_frames", type=int, default=8, help="Số frame trích xuất từ video")
    parser.add_argument("--image_size", type=int, default=224, help="Kích thước ảnh đầu vào")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Thiết bị để chạy model")
    parser.add_argument("--debug", action="store_true", help="Chế độ debug với nhiều thông tin hơn")
    return parser.parse_args()

def load_video(video_path, n_frames=8, image_size=224, device="cuda"):
    """Tải video và trích xuất frames."""
    logger.info(f"Đang tải video từ {video_path}")
    
    try:
        # Sử dụng Decord để đọc video
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(video_path)
        
        # Lấy tổng số frame và FPS
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        logger.info(f"Video có {total_frames} frames, {fps} FPS")
        
        # Chọn n_frames đều đặn trong toàn bộ video
        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        frames = vr.get_batch(indices)
        
        # Chuyển đổi từ (T, H, W, C) sang (T, C, H, W)
        frames = frames.permute(0, 3, 1, 2).to(device)
        
        # Thêm chiều batch
        frames = frames.unsqueeze(0)  # (1, T, C, H, W)
        
        logger.info(f"Đã trích xuất {n_frames} frames với shape {frames.shape}")
        return frames
    
    except Exception as e:
        logger.error(f"Lỗi khi tải video: {e}")
        raise

def main():
    args = parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Tạo config model
    logger.info(f"Đang khởi tạo model BLIP2 TimeSformer với loại {args.model_type}")
    
    device = torch.device(args.device)
    
    # Tạo cấu hình model
    model_config = {
        "arch": "blip2_timesformer",
        "model_type": args.model_type,
        "num_frames": args.n_frames,
    }
    
    # Thêm checkpoint nếu được cung cấp
    if args.checkpoint:
        model_config["pretrained"] = args.checkpoint
    
    cfg = OmegaConf.create({"model": model_config})
    
    # Đăng ký cấu hình video processor
    video_processor_cfg = {
        "n_frms": args.n_frames,
        "image_size": args.image_size,
    }
    
    # Tải model
    model = registry.get_model_class("blip2_timesformer").from_config(cfg.model)
    model = model.to(device)
    model.eval()
    
    # Tạo video processor
    processor = PrepareTimeSformerVideo(
        n_frms=args.n_frames,
        image_size=args.image_size,
    )
    
    # Tải video
    try:
        video_tensor = load_video(
            args.video, 
            n_frames=args.n_frames, 
            image_size=args.image_size,
            device=device
        )
        
        # Sinh caption
        logger.info("Đang sinh caption...")
        with torch.no_grad():
            caption = model.generate(
                video_tensor,
                use_nucleus_sampling=args.use_nucleus_sampling,
                num_beams=args.num_beams,
                max_length=args.max_length,
                min_length=args.min_length,
                repetition_penalty=args.repetition_penalty,
                length_penalty=args.length_penalty,
                top_p=args.top_p,
                temperature=args.temperature,
            )
        
        # In kết quả
        logger.info("Kết quả:")
        logger.info(f"Caption: {caption[0]}")
        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình chạy inference: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 