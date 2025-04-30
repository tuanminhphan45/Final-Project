#!/usr/bin/env python3
import os
import torch
import logging
import argparse
from lavis.processors.alpro_processors import AlproVideoEvalProcessor
from lavis.models.blip2_models.blip2_timesformer import Blip2TimeSformer

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Đường dẫn đến file video")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn đến file .pth checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Thiết bị: {device}")

    # 1. Khởi tạo mô hình
    model = Blip2TimeSformer(
        vit_model="timesformer",
        img_size=224,
        num_frames=8,
        drop_path_rate=0.1,
        use_grad_checkpointing=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=64,
        cross_attention_freq=2,
        embed_dim=768,
        max_txt_len=32,
    )

    # 2. Load checkpoint
    logger.info(f"Loading checkpoint từ {args.checkpoint} …")
    ckpt = torch.load(args.checkpoint, map_location=device)
    # Hỗ trợ nhiều kiểu key phổ biến
    state = (
        ckpt.get("model_state")
        or ckpt.get("state_dict")
        or ckpt.get("model")
        or ckpt
    )
    model.load_state_dict(state, strict=False)
    logger.info("Load checkpoint xong")

    # 3. Move & half
    model = model.to(device).half()
    model.eval()

    # 4. Chuẩn bị video processor
    video_processor = AlproVideoEvalProcessor(image_size=224, n_frms=8, full_video=True)

    # 5. Xử lý video
    processed = video_processor(args.video)                         # [C,T,H,W]
    batch = processed.unsqueeze(0).to(device).half()                # [1,C,T,H,W]
    logger.info(f"Video tensor: {batch.shape}, dtype={batch.dtype}")

    # 6. Generate
    with torch.no_grad(), torch.cuda.amp.autocast():
        caption = model.generate(
            {"video": batch},
            use_nucleus_sampling=True,
            num_beams=1,
            max_length=30,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.15,
        )[0]

    # 7. In ra
    print("\n--- KẾT QUẢ ---")
    print(f"Video: {args.video}")
    print(f"Caption: {caption}")

if __name__ == "__main__":
    main()
