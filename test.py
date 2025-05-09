#!/usr/bin/env python3
import os
import torch
import logging
import argparse
from lavis.processors.alpro_processors import AlproVideoEvalProcessor
from lavis.models.blip2_models.blip2_timesformer import Blip2TimeSformer
import torch.nn.functional as F

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
        num_frames=16,
        drop_path_rate=0.1,
        use_grad_checkpointing=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=128,
        cross_attention_freq=2,
        embed_dim=768,
        max_txt_len=32,
    )

    # 2. Load checkpoint
    logger.info(f"Loading checkpoint từ {args.checkpoint} …")
    try:
        ckpt = torch.load(args.checkpoint, map_location=device)
        logger.info(f"Cấu trúc checkpoint: {list(ckpt.keys()) if isinstance(ckpt, dict) else 'không phải dict'}")
        
        # Hỗ trợ nhiều kiểu key phổ biến
        if "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
        
        # Xử lý đặc biệt cho keys của TimeSformer
        spatial_embed_key = "visual_encoder.model.pos_embed"
        temporal_embed_key = "visual_encoder.model.time_embed"
        
        # Lấy thông tin số frames và patches từ mô hình
        num_patches = (224 // 16) ** 2  # image_size // patch_size
        num_frames = 16  # n_frms như đã khai báo
        
        # Xử lý embedding cho TimeSformer
        if spatial_embed_key in state_dict and num_patches + 1 != state_dict[spatial_embed_key].size(1):
            logger.info(f"Điều chỉnh spatial embedding từ {state_dict[spatial_embed_key].size(1)} thành {num_patches + 1}")
            pos_embed = state_dict[spatial_embed_key]
            cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
            new_pos_embed = F.interpolate(other_pos_embed, size=(num_patches), mode="nearest")
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            state_dict[spatial_embed_key] = new_pos_embed
        
        if temporal_embed_key in state_dict and num_frames != state_dict[temporal_embed_key].size(1):
            logger.info(f"Điều chỉnh temporal embedding từ {state_dict[temporal_embed_key].size(1)} thành {num_frames}")
            time_embed = state_dict[temporal_embed_key].transpose(1, 2)
            new_time_embed = F.interpolate(time_embed, size=(num_frames), mode="nearest")
            state_dict[temporal_embed_key] = new_time_embed.transpose(1, 2)
        
        # Load checkpoint với strict=False
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Các keys thiếu: {len(missing)} keys")
            logger.warning(f"Một số keys thiếu: {missing[:10]}...")
        if unexpected:
            logger.warning(f"Các keys không mong đợi: {len(unexpected)} keys")
            logger.warning(f"Một số keys không mong đợi: {unexpected[:10]}...")
            
        logger.info("Load checkpoint xong (với strict=False)")
    except Exception as e:
        logger.error(f"Lỗi khi load checkpoint: {str(e)}")
        # Print stack trace
        import traceback
        logger.error(traceback.format_exc())
        # Không dừng chương trình, tiếp tục với mô hình không có checkpoint
        logger.warning("Tiếp tục mà không load checkpoint")

    # 3. Move & half
    model = model.to(device).half()
    model.eval()

    # 4. Chuẩn bị video processor
    video_processor = AlproVideoEvalProcessor(image_size=224, n_frms=16, full_video=True)

    # 5. Xử lý video
    processed = video_processor(args.video)                         # [C,T,H,W]
    batch = processed.unsqueeze(0).to(device).half()                # [1,C,T,H,W]
    logger.info(f"Video tensor: {batch.shape}, dtype={batch.dtype}")

    # 6. Generate
    with torch.no_grad(), torch.cuda.amp.autocast():
        caption = model.generate(
            {"video": batch},
            use_nucleus_sampling=False,
            num_beams=1,
            max_length=50,
            min_length=10,
            top_p=0.9,
            repetition_penalty=1.15,
        )[0]

    # 7. In ra
    print("\n--- KẾT QUẢ ---")
    print(f"Video: {args.video}")
    print(f"Caption: {caption}")

if __name__ == "__main__":
    main()
