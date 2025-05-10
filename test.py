#!/usr/bin/env python3
import os
import torch
import logging
import argparse
import torch.nn.functional as F
from lavis.processors.alpro_processors import AlproVideoEvalProcessor
from lavis.models.blip2_models.blip2_timesformer import Blip2TimeSformer

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Đường dẫn đến file video")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn đến file .pth checkpoint")
    parser.add_argument("--timesformer_checkpoint", type=str, 
                        default="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/ALPRO/alpro_pretrain.pt",
                        help="URL hoặc đường dẫn đến file pretrained weights cho TimeSformer")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Thiết bị: {device}")

    # 1. Khởi tạo mô hình với timesformer_weight_path
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
        timesformer_weight_path=args.timesformer_checkpoint,  # Load TimeSformer weights trong constructor
    )

    # 2. Load checkpoint đã train
    logger.info(f"Loading checkpoint từ {args.checkpoint}")
    try:
        # Sử dụng phương thức load_from_pretrained
        model.load_from_pretrained(args.checkpoint)
        logger.info("Đã load checkpoint thành công")
    except Exception as e:
        logger.error(f"Lỗi khi load checkpoint: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # 3. Move model to device và chuyển sang half precision
    model = model.to(device).half()
    model.eval()

    # 4. Chuẩn bị video processor
    video_processor = AlproVideoEvalProcessor(image_size=224, n_frms=16, full_video=True)

    # 5. Xử lý video
    processed = video_processor(args.video)                         # [C,T,H,W]
    batch = processed.unsqueeze(0).to(device).half()                # [1,C,T,H,W]
    logger.info(f"Video tensor: {batch.shape}, dtype={batch.dtype}")

    # 6. Generate caption
    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
        caption = model.generate(
            {"video": batch},
            use_nucleus_sampling=False,
            num_beams=3,
            max_length=50,
            min_length=10,
            top_p=0.9,
            repetition_penalty=1.15,
        )[0]

    # 7. In kết quả
    print("\n--- KẾT QUẢ ---")
    print(f"Video: {args.video}")
    print(f"Caption: {caption}")

if __name__ == "__main__":
    main()
