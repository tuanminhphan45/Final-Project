#!/bin/bash
export MASTER_PORT=55555
export MASTER_ADDR=localhost
export CUDA_VISIBLE_DEVICES=1,2,3

export LAVIS_CACHE_ROOT="/storage/student10/vidcaption/LAVIS/cache"
cd /storage/student10/vidcaption/LAVIS
export PYTHONPATH=/storage/student10/vidcaption/LAVIS:$PYTHONPATH

# Tạo file test để kiểm tra việc load weights
cat > test_weights_loading.py << 'EOF'
import os
import sys
import torch
import yaml
import logging
from lavis.models import load_model_and_preprocess
from omegaconf import OmegaConf

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("weight_check")

def check_weights_loading():
    logger.info("===== KIỂM TRA VIỆC LOAD WEIGHTS CHO BLIP2 TIMESFORMER =====")
    
    # Load file config
    config_path = "lavis/projects/blip2_timesformer/eval/caption_coco_timesformer_eval.yaml"
    logger.info(f"Đang load config từ {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # In thông tin về pretrained weights trong config
    timesformer_pretrained = config["model"].get("timesformer_pretrained", "không có")
    qformer_pretrained = config["model"].get("qformer_pretrained", "không có")
    eval_ckpt_path = config["run"].get("eval_ckpt_path", "không có")
    
    logger.info(f"Config - TimeSformer pretrained: {timesformer_pretrained}")
    logger.info(f"Config - QFormer pretrained: {qformer_pretrained}")
    logger.info(f"Config - Checkpoint đánh giá: {eval_ckpt_path}")
    
    # Thêm tham số qformer_pretrained nếu chưa có
    if "qformer_pretrained" not in config["model"] and eval_ckpt_path != "không có":
        logger.info(f"Thêm tham số qformer_pretrained = {eval_ckpt_path}")
        config["model"]["qformer_pretrained"] = eval_ckpt_path
    
    # Load model
    logger.info("Đang load model...")
    try:
        model_config = OmegaConf.create(config["model"])
        model_cls = "blip2_timesformer"
        model, vis_processors, txt_processors = load_model_and_preprocess(
            model_cls, model_config
        )
        logger.info("✓ Load model thành công")
        
        # Kiểm tra TimeSformer weights
        if hasattr(model, "visual_encoder") and hasattr(model.visual_encoder, "model"):
            logger.info(f"TimeSformer pos_embed shape: {model.visual_encoder.model.pos_embed.shape}")
            if hasattr(model.visual_encoder.model, "time_embed"):
                logger.info(f"TimeSformer time_embed shape: {model.visual_encoder.model.time_embed.shape}")
                logger.info("✓ TimeSformer weights đã được load")
            else:
                logger.warning("⚠️ TimeSformer time_embed không tồn tại")
        else:
            logger.error("✗ TimeSformer weights không được load đúng")
        
        # Kiểm tra QFormer weights
        if hasattr(model, "Qformer") and hasattr(model.Qformer, "bert"):
            logger.info(f"QFormer config hidden size: {model.Qformer.config.hidden_size}")
            sample_weight = model.Qformer.bert.encoder.layer[0].attention.self.query.weight.mean().item()
            logger.info(f"QFormer sample weight mean: {sample_weight:.6f}")
            if abs(sample_weight) > 1e-4:
                logger.info("✓ QFormer weights có khả năng đã được load (các giá trị khác 0)")
            else:
                logger.warning("⚠️ QFormer weights có thể chưa được load (các giá trị gần 0)")
        else:
            logger.error("✗ QFormer không được khởi tạo đúng")
            
    except Exception as e:
        logger.error(f"Lỗi khi load model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("===== KẾT THÚC KIỂM TRA =====")

if __name__ == "__main__":
    check_weights_loading()
EOF

# Chạy script kiểm tra
echo "Đang chạy script kiểm tra load weights..."
python test_weights_loading.py

# Chạy script evaluate như bình thường
echo "Tiếp tục với quá trình evaluate..."
python -m torch.distributed.run \
    --nproc_per_node=3 \
    --master_port=${MASTER_PORT} \
    evaluate.py \
    --cfg-path lavis/projects/blip2_timesformer/eval/caption_coco_timesformer_eval.yaml 