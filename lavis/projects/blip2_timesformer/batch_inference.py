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
import glob
import json
import time
import decord
import numpy as np
from tqdm import tqdm
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
logger = logging.getLogger("blip2_timesformer_batch_inference")

def parse_args():
    parser = argparse.ArgumentParser(description="BLIP2 TimeSformer Batch Inference")
    parser.add_argument("--video_dir", required=True, help="Thư mục chứa các video")
    parser.add_argument("--output_file", required=True, help="File đầu ra để lưu kết quả (định dạng JSON)")
    parser.add_argument("--video_ext", default="mp4", help="Định dạng video cần xử lý")
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
    parser.add_argument("--batch_size", type=int, default=1, help="Kích thước batch (số video xử lý đồng thời)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Thiết bị để chạy model")
    parser.add_argument("--debug", action="store_true", help="Chế độ debug với nhiều thông tin hơn")
    return parser.parse_args()

def load_video(video_path, n_frames=8, image_size=224):
    """Tải video và trích xuất frames."""
    try:
        # Sử dụng Decord để đọc video
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(video_path)
        
        # Lấy tổng số frame và FPS
        total_frames = len(vr)
        
        # Chọn n_frames đều đặn trong toàn bộ video
        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        frames = vr.get_batch(indices)
        
        # Chuyển đổi từ (T, H, W, C) sang (T, C, H, W)
        frames = frames.permute(0, 3, 1, 2)
        
        return frames
    
    except Exception as e:
        logger.error(f"Lỗi khi tải video {video_path}: {e}")
        return None

def collect_videos(video_dir, ext="mp4"):
    """Thu thập đường dẫn của tất cả các video trong thư mục."""
    pattern = os.path.join(video_dir, f"*.{ext}")
    video_paths = glob.glob(pattern)
    logger.info(f"Tìm thấy {len(video_paths)} video với định dạng .{ext}")
    return video_paths

def process_batch(model, video_paths, args, device):
    """Xử lý một batch video và tạo caption."""
    results = []
    
    # Tải và xử lý các video
    valid_videos = []
    valid_paths = []
    
    for video_path in video_paths:
        video_tensor = load_video(video_path, args.n_frames, args.image_size)
        if video_tensor is not None:
            valid_videos.append(video_tensor)
            valid_paths.append(video_path)
    
    if not valid_videos:
        return results
    
    # Kết hợp các video thành một batch
    batch_videos = torch.stack(valid_videos).to(device)
    
    # Tạo caption cho các video
    with torch.no_grad():
        captions = model.generate(
            batch_videos,
            use_nucleus_sampling=args.use_nucleus_sampling,
            num_beams=args.num_beams,
            max_length=args.max_length,
            min_length=args.min_length,
            repetition_penalty=args.repetition_penalty,
            length_penalty=args.length_penalty,
            top_p=args.top_p,
            temperature=args.temperature,
        )
    
    # Lưu kết quả
    for video_path, caption in zip(valid_paths, captions):
        video_name = os.path.basename(video_path)
        results.append({
            "video_path": video_path,
            "video_name": video_name,
            "caption": caption
        })
    
    return results

def main():
    args = parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Tạo thư mục đầu ra nếu cần
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    device = torch.device(args.device)
    
    # Tạo cấu hình model
    logger.info(f"Đang khởi tạo model BLIP2 TimeSformer với loại {args.model_type}")
    
    model_config = {
        "arch": "blip2_timesformer",
        "model_type": args.model_type,
        "num_frames": args.n_frames,
    }
    
    # Thêm checkpoint nếu được cung cấp
    if args.checkpoint:
        model_config["pretrained"] = args.checkpoint
        logger.info(f"Sử dụng checkpoint từ {args.checkpoint}")
    
    cfg = OmegaConf.create({"model": model_config})
    
    # Tải model
    model = registry.get_model_class("blip2_timesformer").from_config(cfg.model)
    model = model.to(device)
    model.eval()
    
    # Thu thập tất cả các video
    video_paths = collect_videos(args.video_dir, args.video_ext)
    
    if not video_paths:
        logger.error(f"Không tìm thấy video nào trong thư mục {args.video_dir} với định dạng {args.video_ext}")
        return
    
    # Xử lý theo batch
    all_results = []
    total_batches = (len(video_paths) + args.batch_size - 1) // args.batch_size
    
    start_time = time.time()
    
    for i in tqdm(range(0, len(video_paths), args.batch_size), desc="Xử lý video", total=total_batches):
        batch_paths = video_paths[i:i+args.batch_size]
        batch_results = process_batch(model, batch_paths, args, device)
        all_results.extend(batch_results)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Lưu kết quả
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Đã xử lý {len(all_results)} video trong {processing_time:.2f} giây")
    logger.info(f"Kết quả đã được lưu vào {args.output_file}")

if __name__ == "__main__":
    main() 