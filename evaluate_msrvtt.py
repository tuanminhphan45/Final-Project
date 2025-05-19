#!/usr/bin/env python3
import os
import json
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from lavis.processors.alpro_processors import AlproVideoEvalProcessor
from lavis.models.blip2_models.blip2_timesformer import Blip2TimeSformer

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

class MSRVTTDataset(Dataset):
    def __init__(self, video_dir, annotations, video_processor):
        self.video_dir = video_dir
        self.annotations = annotations
        self.video_processor = video_processor
        self.video_ids = list(annotations.keys())
        
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        captions = self.annotations[video_id]
        
        # Đường dẫn video có thể có hoặc không có đuôi .mp4
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            # Thử lại nếu file không tồn tại (có thể video_id đã có đuôi .mp4)
            video_path = os.path.join(self.video_dir, video_id)
            if not os.path.exists(video_path):
                logger.error(f"Không tìm thấy video: {video_id}")
                return None
        
        try:
            processed = self.video_processor(video_path)
            return {
                'video_id': video_id,
                'video': processed,
                'captions': captions
            }
        except Exception as e:
            logger.error(f"Lỗi khi xử lý video {video_id}: {str(e)}")
            return None

def setup_distributed():
    """Thiết lập distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        # Map GPU index to 1,2,3
        gpu = gpu + 1
    else:
        rank = 0
        world_size = 1
        gpu = 1  # Default to GPU 1

    # Kiểm tra GPU có tồn tại không
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA không khả dụng")
    
    if gpu >= torch.cuda.device_count():
        raise RuntimeError(f"GPU {gpu} không tồn tại. Số GPU khả dụng: {torch.cuda.device_count()}")

    # Set device
    torch.cuda.set_device(gpu)
    
    # Khởi tạo process group
    try:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo process group: {str(e)}")
        raise

    return rank, world_size, gpu

def load_msrvtt_annotations(annotation_file):
    """Load MSR-VTT annotations"""
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # In thông tin về cấu trúc file
    if isinstance(annotations, list) and len(annotations) > 0:
        first_item = annotations[0]
        logger.info(f"Cấu trúc annotation: {list(first_item.keys())}")
        logger.info(f"Số lượng annotations: {len(annotations)}")
    else:
        logger.info(f"Annotations không phải dạng list hoặc rỗng. Type: {type(annotations)}")
    
    return annotations

def prepare_references(annotations):
    """Chuẩn bị references cho mỗi video"""
    video_to_captions = {}
    for item in annotations:
        # Sử dụng trường 'video' thay vì 'video_id'
        video_id = item.get('video', item.get('image_id'))
        
        # Loại bỏ đuôi .mp4 nếu có
        if video_id.endswith('.mp4'):
            video_id = video_id[:-4]
            
        if video_id not in video_to_captions:
            video_to_captions[video_id] = []
        video_to_captions[video_id].append(item['caption'])
    
    # Log thông tin số lượng video và caption
    logger.info(f"Đã load {len(video_to_captions)} videos với tổng cộng {sum(len(caps) for caps in video_to_captions.values())} captions")
    
    return video_to_captions

def compute_metrics(predictions, references):
    """Tính toán các metrics sử dụng pycocoevalcap"""
    # Chuẩn bị dữ liệu cho evaluation
    gts = {}
    res = {}
    for i, (pred, refs) in enumerate(zip(predictions, references)):
        gts[i] = refs
        res[i] = [pred]

    # Khởi tạo scorers
    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE-L"),
        (Cider(), "CIDEr")
    ]

    # Tính toán metrics
    metrics = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for sc, m in zip(score, method):
                metrics[m] = sc
        else:
            metrics[method] = score

    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn đến file .pth checkpoint đã training")
    parser.add_argument("--timesformer_checkpoint", type=str, 
                        default="/home/student10/.cache/torch/hub/checkpoints/alpro_msrvtt_qa.pth",
                        help="Đường dẫn đến TimeSformer weights")
    parser.add_argument("--video_dir", type=str, default="datasets/msrvtt/videos",
                        help="Thư mục chứa video files")
    parser.add_argument("--annotation_file", type=str, default="datasets/msrvtt/annotations/cap_test.json",
                        help="File annotation cho test set")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                        help="File để lưu kết quả evaluation")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size cho mỗi GPU")
    args = parser.parse_args()

    # Thiết lập distributed
    rank, world_size, gpu = setup_distributed()
    device = torch.device(f"cuda:{gpu}")

    # 1. Load annotations
    if rank == 0:
        logger.info("Loading annotations...")
    annotations = load_msrvtt_annotations(args.annotation_file)
    video_to_captions = prepare_references(annotations)
    
    # 2. Khởi tạo model
    model = Blip2TimeSformer(
        vit_model="timesformer",
        img_size=224,
        num_frames=8,
        drop_path_rate=0.1,
        use_grad_checkpointing=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=128,
        cross_attention_freq=2,
        embed_dim=768,
        max_txt_len=32,
    )

    # 3. Load checkpoint
    if rank == 0:
        logger.info(f"Loading checkpoint {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        
        # Kiểm tra cấu trúc checkpoint
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
            
        # Lọc bỏ các weights của visual_encoder để tránh ghi đè lên TimeSformer đã được load
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if not key.startswith("visual_encoder"):
                filtered_state_dict[key] = value
        
        # Load state dict đã lọc vào model
        msg = model.load_state_dict(filtered_state_dict, strict=False)
        if rank == 0:
            logger.info(f"Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
            if msg.missing_keys:
                if len(msg.missing_keys) > 5:
                    logger.info(f"missing keys: {msg.missing_keys[:5]}...")
                else:
                    logger.info(f"Missing keys: {msg.missing_keys}")
            
            # Kiểm tra số lượng parameters
            visual_encoder_params = len([name for name, _ in model.named_parameters() if "visual_encoder" in name])
            logger.info(f"parameters của visual_encoder: {visual_encoder_params}")
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"total parameters của model: {total_params}")
            
    except Exception as e:
        logger.error(f"Lỗi khi load checkpoint: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        dist.destroy_process_group()
        exit(1)

    model = model.to(device).half()
    model = DDP(model, device_ids=[gpu])

    # 4. Chuẩn bị dataset và dataloader
    video_processor = AlproVideoEvalProcessor(image_size=224, n_frms=8, full_video=True)
    dataset = MSRVTTDataset(args.video_dir, video_to_captions, video_processor)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    # 5. Evaluation
    if rank == 0:
        logger.info("Bắt đầu evaluation...")
    
    model.eval()
    predictions = []
    references = []
    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, disable=rank != 0):
            if batch is None:
                continue
                
            videos = batch['video'].to(device).half()
            video_ids = batch['video_id']
            captions = batch['captions']

            # Generate captions
            with torch.amp.autocast(device_type='cuda'):
                generated_captions = model.module.generate(
                    {"video": videos},
                    use_nucleus_sampling=False,
                    num_beams=3,
                    max_length=30,
                    min_length=10,
                    top_p=0.9,
                    repetition_penalty=1.15,
                )

            # Thu thập kết quả
            for video_id, pred_caption, ref_captions in zip(video_ids, generated_captions, captions):
                predictions.append(pred_caption)
                references.append(ref_captions)
                results.append({
                    'video_id': video_id,
                    'predicted_caption': pred_caption,
                    'reference_captions': ref_captions
                })

    # 6. Gather results từ tất cả các GPU
    all_predictions = [None for _ in range(world_size)]
    all_references = [None for _ in range(world_size)]
    all_results = [None for _ in range(world_size)]
    
    dist.all_gather_object(all_predictions, predictions)
    dist.all_gather_object(all_references, references)
    dist.all_gather_object(all_results, results)

    if rank == 0:
        # Flatten results
        all_predictions = [p for preds in all_predictions for p in preds]
        all_references = [r for refs in all_references for r in refs]
        all_results = [r for res in all_results for r in res]

        # 7. Tính toán metrics
        logger.info("Tính toán metrics...")
        metrics = compute_metrics(all_predictions, all_references)
        
        # 8. Lưu kết quả
        output = {
            'metrics': metrics,
            'results': all_results
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Kết quả evaluation đã được lưu vào {args.output_file}")
        logger.info("Metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main() 