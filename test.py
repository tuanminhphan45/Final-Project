#!/usr/bin/env python3
import os
import torch
import logging
import argparse
import torch.nn.functional as F
from lavis.processors.alpro_processors import AlproVideoEvalProcessor
from lavis.models.blip2_models.blip2_timesformer import Blip2TimeSformer
from lavis.models.timesformer.helpers import load_state_dict

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Đường dẫn đến file video")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn đến file .pth checkpoint")
    parser.add_argument("--timesformer_checkpoint", type=str, default="", help="Đường dẫn đến pretrained weights cho TimeSformer")
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

    # 1.1 Load pretrained weights cho TimeSformer nếu có
    if args.timesformer_checkpoint and os.path.exists(args.timesformer_checkpoint):
        logger.info(f"Loading TimeSformer pretrained weights từ {args.timesformer_checkpoint}")
        try:
            # Tính số patches và frames
            num_patches = (224 // 16) ** 2  # image_size // patch_size
            num_frames = 16  # n_frms như đã khai báo
            attention_type = "divided_space_time"

            # Load pretrained weights cho TimeSformer
            ts_state_dict = load_state_dict(args.timesformer_checkpoint)
            
            # Xử lý embedding cho TimeSformer
            if "pos_embed" in ts_state_dict and num_patches + 1 != ts_state_dict["pos_embed"].size(1):
                logger.info(f"Điều chỉnh spatial embedding từ {ts_state_dict['pos_embed'].size(1)} thành {num_patches + 1}")
                pos_embed = ts_state_dict["pos_embed"]
                cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
                other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
                new_pos_embed = F.interpolate(other_pos_embed, size=(num_patches), mode="nearest")
                new_pos_embed = new_pos_embed.transpose(1, 2)
                new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
                ts_state_dict["pos_embed"] = new_pos_embed
            
            if "time_embed" in ts_state_dict and num_frames != ts_state_dict["time_embed"].size(1):
                logger.info(f"Điều chỉnh temporal embedding từ {ts_state_dict['time_embed'].size(1)} thành {num_frames}")
                time_embed = ts_state_dict["time_embed"].transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(num_frames), mode="nearest")
                ts_state_dict["time_embed"] = new_time_embed.transpose(1, 2)
            
            # Thêm tiền tố "visual_encoder.model." vào tất cả keys
            new_ts_state_dict = {}
            for key, value in ts_state_dict.items():
                new_ts_state_dict[f"visual_encoder.model.{key}"] = value
            
            # Load weights vào mô hình
            missing, unexpected = model.load_state_dict(new_ts_state_dict, strict=False)
            logger.info(f"TimeSformer loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        except Exception as e:
            logger.error(f"Lỗi khi load TimeSformer weights: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

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
        
        # Lọc bỏ các weights của visual_encoder nếu đã load pretrained weights
        if args.timesformer_checkpoint and os.path.exists(args.timesformer_checkpoint):
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if not key.startswith("visual_encoder"):
                    filtered_state_dict[key] = value
            state_dict = filtered_state_dict
            logger.info(f"Đã lọc bỏ các weights của visual_encoder từ checkpoint chính")
        
        # Load checkpoint với strict=False
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Các keys thiếu: {len(missing)} keys")
            logger.warning(f"Một số keys thiếu: {missing[:10]}...")
        if unexpected:
            logger.warning(f"Các keys không mong đợi: {len(unexpected)} keys")
            logger.warning(f"Một số keys không mong đợi: {unexpected[:10]}...")
            
        logger.info("Load checkpoint chính xong (với strict=False)")
    except Exception as e:
        logger.error(f"Lỗi khi load checkpoint: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
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
