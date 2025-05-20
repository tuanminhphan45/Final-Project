#!/usr/bin/env python3
import os
import json
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
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
        
        # Kiểm tra duplicate video_ids
        unique_video_ids = set(self.video_ids)
        if len(unique_video_ids) != len(self.video_ids):
            logger.warning(f"Duplicate video_ids detected! {len(self.video_ids)} ids, but only {len(unique_video_ids)} unique ids")
            # Đảm bảo không có video_ids trùng lặp
            self.video_ids = list(unique_video_ids)
            logger.info(f"Removed duplicate video_ids, {len(self.video_ids)} videos remaining")
        else:
            logger.info(f"No duplicate video_ids detected. {len(self.video_ids)} videos")
            
        # Kiểm tra số lượng caption trung bình
        avg_caps = sum(len(caps) for caps in annotations.values()) / len(annotations) if annotations else 0
        logger.info(f"In MSRVTTDataset: Average captions per video: {avg_caps:.2f}")
        
        # Kiểm tra số caption của một số video cụ thể
        if "video7010" in self.annotations:
            logger.info(f"In MSRVTTDataset: Video7010 has {len(self.annotations['video7010'])} captions")
            
        # Lưu video_id đã được xử lý để tránh duplicate
        self.processed_videos = set()
        
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        
        # Kiểm tra nếu video đã được xử lý trước đó
        if video_id in self.processed_videos:
            logger.warning(f"Video {video_id} was processed before! Possible duplicate.")
        
        # Đánh dấu video đã xử lý
        self.processed_videos.add(video_id)
        
        captions = self.annotations[video_id]
        
        # Kiểm tra số lượng caption đang được trả về
        if idx < 3:  # Chỉ log cho 3 video đầu tiên
            logger.info(f"__getitem__: Video {video_id} returning {len(captions)} captions")
        
        # Đường dẫn video có thể có hoặc không có đuôi .mp4
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            # Thử lại nếu file không tồn tại (có thể video_id đã có đuôi .mp4)
            video_path = os.path.join(self.video_dir, video_id)
            if not os.path.exists(video_path):
                logger.error(f"Video not found: {video_id}")
                return None
        
        try:
            processed = self.video_processor(video_path)
            return {
                'video_id': video_id,
                'video': processed,
                'captions': captions
            }
        except Exception as e:
            logger.error(f"Error processing video {video_id}: {str(e)}")
            return None

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
    
    # Đếm số lượng annotation cho mỗi video để kiểm tra
    video_count = {}
    video_ids_seen = set()
    
    for item in annotations:
        # Sử dụng trường 'video' hoặc 'image_id' làm video_id
        video_id = item.get('video', item.get('image_id'))
        
        # Loại bỏ đuôi .mp4 nếu có
        if video_id and isinstance(video_id, str) and video_id.endswith('.mp4'):
            video_id = video_id[:-4]
            
        if not video_id:
            continue
        
        # Kiểm tra xem video_id có trùng lặp không
        if video_id in video_ids_seen:
            logger.info(f"Video ID already exists in annotations: {video_id}")
        else:
            video_ids_seen.add(video_id)
            
        # Đếm số lượng annotation
        video_count[video_id] = video_count.get(video_id, 0) + 1
            
        if video_id not in video_to_captions:
            video_to_captions[video_id] = []
            
        # Kiểm tra xem caption có tồn tại không
        if 'caption' in item and item['caption']:
            video_to_captions[video_id].append(item['caption'])
    
    # Log thông tin chi tiết
    logger.info(f"Total videos in file: {len(video_to_captions)}")
    logger.info(f"Total captions in file: {sum(len(caps) for caps in video_to_captions.values())}")
    
    # In ra phân phối số lượng caption
    caption_counts = {}
    for vid, caps in video_to_captions.items():
        count = len(caps)
        caption_counts[count] = caption_counts.get(count, 0) + 1
    
    logger.info("Caption count distribution:")
    for count, num_videos in sorted(caption_counts.items()):
        logger.info(f"  {count} captions: {num_videos} videos")
    
    # Kiểm tra một vài video cụ thể
    sample_videos = list(video_to_captions.keys())[:3]
    for vid in sample_videos:
        logger.info(f"Video {vid} has {len(video_to_captions[vid])} captions")
        
    # Kiểm tra số caption có trong dataset vs số caption được thu thập
    logger.info(f"Comparing annotation counts vs collected captions:")
    for vid in sample_videos:
        logger.info(f"Video {vid}: {video_count.get(vid, 0)} annotations, {len(video_to_captions[vid])} captions")
    
    return video_to_captions

def compute_metrics(predictions, references):
    """Tính toán các metrics sử dụng pycocoevalcap"""
    # In thông tin để debug
    if references and len(references) > 0:
        logger.info(f"Số lượng video để đánh giá: {len(predictions)}")
        logger.info(f"Số lượng references cho video đầu tiên: {len(references[0])}")
        
    # Chuẩn bị dữ liệu cho evaluation
    gts = {}
    res = {}
    for i, (pred, refs) in enumerate(zip(predictions, references)):
        gts[i] = refs
        res[i] = [pred]
    
    # In thêm thông tin về dữ liệu đã chuẩn bị
    if gts and len(gts) > 0:
        first_key = list(gts.keys())[0]
        logger.info(f"Dữ liệu đánh giá cho video đầu tiên (id={first_key}):")
        logger.info(f"  - Số lượng references: {len(gts[first_key])}")
        logger.info(f"  - Prediction: {res[first_key][0]}")

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

def custom_collate_fn(batch):
    """
    Hàm collate tùy chỉnh để giữ nguyên số lượng caption cho mỗi video và kiểm tra input
    """
    # Loại bỏ các None items (videos không tìm thấy)
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    # Kết hợp các item thành batch
    video_ids = [item['video_id'] for item in batch]
    videos = torch.stack([item['video'] for item in batch])
    captions = [item['captions'] for item in batch]
    
    # Kiểm tra và log chi tiết
    caption_lengths = [len(caps) for caps in captions]
    logger.info(f"custom_collate_fn: Batch with {len(videos)} videos: {video_ids}")
    logger.info(f"custom_collate_fn: Video tensor shape: {videos.shape}")
    
    return {
        'video_id': video_ids,
        'video': videos,
        'captions': captions
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--timesformer_checkpoint", type=str, 
                        default="/home/student10/.cache/torch/hub/checkpoints/alpro_msrvtt_qa.pth",
                        help="Path to TimeSformer weights")
    parser.add_argument("--video_dir", type=str, default="datasets/msrvtt/videos",
                        help="Directory containing video files")
    parser.add_argument("--annotation_file", type=str, default="datasets/msrvtt/annotations/cap_test.json",
                        help="Annotation file for test set")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                        help="File to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    args = parser.parse_args()

    # Kiểm tra GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    if args.gpu_id >= torch.cuda.device_count():
        raise RuntimeError(f"GPU {args.gpu_id} doesn't exist. Available GPUs: {torch.cuda.device_count()}")

    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(args.gpu_id)
    
    logger.info(f"Using GPU: {torch.cuda.get_device_name(args.gpu_id)}")

    # 1. Load annotations
    logger.info("Loading annotations...")
    annotations = load_msrvtt_annotations(args.annotation_file)
    video_to_captions = prepare_references(annotations)
    
    # 2. Initialize model
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
    logger.info(f"Loading checkpoint {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        
        # Check checkpoint structure
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
            
        # Filter out visual_encoder weights to avoid overriding TimeSformer
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if not key.startswith("visual_encoder"):
                filtered_state_dict[key] = value
        
        # Load filtered state dict into model
        msg = model.load_state_dict(filtered_state_dict, strict=False)
        logger.info(f"Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
        if msg.missing_keys:
            if len(msg.missing_keys) > 5:
                logger.info(f"missing keys: {msg.missing_keys[:5]}...")
            else:
                logger.info(f"Missing keys: {msg.missing_keys}")
        
        # Check parameter counts
        visual_encoder_params = len([name for name, _ in model.named_parameters() if "visual_encoder" in name])
        logger.info(f"visual_encoder parameters: {visual_encoder_params}")
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"total model parameters: {total_params}")
            
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        exit(1)

    model = model.to(device).half()
    model.eval()

    # 4. Prepare dataset and dataloader
    video_processor = AlproVideoEvalProcessor(image_size=224, n_frms=8, full_video=True)
    dataset = MSRVTTDataset(args.video_dir, video_to_captions, video_processor)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    # 5. Evaluation
    logger.info("Starting evaluation...")
    
    predictions = []
    references = []
    results = []
    
    # Map to store video_id -> predicted_caption to check duplicates
    video_to_prediction = {}

    with torch.no_grad():
        # Configure tqdm to display on a single line
        for batch_idx, batch in enumerate(tqdm(dataloader, ncols=80, leave=True, position=0)):
            if batch is None:
                continue
                
            videos = batch['video'].to(device).half()
            video_ids = batch['video_id']
            captions = batch['captions']

            # Check number of captions for each video
            for vid, caps in zip(video_ids, captions):
                logger.info(f"Video {vid} has {len(caps)} captions for evaluation")

            # Generate captions for EACH video SEPARATELY to avoid confusion
            generated_captions = []
            for i, vid in enumerate(video_ids):
                logger.info(f"Generating caption for video {vid} (index {i} in batch {batch_idx})")
                
                # Get a specific video from the batch
                single_video = videos[i:i+1]  # Create batch size 1
                
                with torch.amp.autocast(device_type='cuda'):
                    # Generate caption for a single video
                    caption = model.generate(
                        {"video": single_video},
                        use_nucleus_sampling=False,
                        num_beams=3,
                        max_length=30,
                        min_length=10,
                        top_p=0.9,
                        repetition_penalty=1.15,
                    )[0]  # Take the first (and only) caption from the results
                    
                    # Check if this video already has a caption
                    if vid in video_to_prediction:
                        logger.warning(f"Video {vid} already has a previous caption: '{video_to_prediction[vid]}'")
                        logger.warning(f"New caption: '{caption}'")
                        
                    # Save caption to check for duplicates
                    video_to_prediction[vid] = caption
                    generated_captions.append(caption)
            
            # Collect results
            for video_id, pred_caption, ref_captions in zip(video_ids, generated_captions, captions):
                # Check number of captions for each video before saving
                logger.info(f"Saving results: Video {video_id} has {len(ref_captions)} reference captions")
                
                predictions.append(pred_caption)
                references.append(ref_captions)
                # Save all captions in results
                results.append({
                    'video_id': video_id,
                    'predicted_caption': pred_caption,
                    'reference_captions': ref_captions,
                    'num_references': len(ref_captions)  # Add reference count for verification
                })

    # Check number of videos with different captions
    logger.info(f"Total videos processed: {len(video_to_prediction)}")
    logger.info(f"Total captions generated: {len(predictions)}")
    
    # 6. Calculate metrics
    logger.info("Calculating metrics...")
    # Check average caption count used
    avg_refs = sum(len(refs) for refs in references) / len(references) if references else 0
    logger.info(f"Average references per video: {avg_refs:.2f}")
    
    # Log some examples
    if references:
        logger.info(f"Example: First video has {len(references[0])} captions")
        
    metrics = compute_metrics(predictions, references)
    
    # 7. Save results
    output = {
        'metrics': metrics,
        'results': results,
        'avg_references_per_video': avg_refs
    }
    
    # Check caption counts in results
    if results:
        caption_counts = [r['num_references'] for r in results]
        logger.info(f"Caption count distribution in results:")
        unique_counts = sorted(set(caption_counts))
        for count in unique_counts:
            num_videos = caption_counts.count(count)
            logger.info(f"  {count} captions: {num_videos} videos")
        
        # Check some specific results
        for i, r in enumerate(results[:3]):
            logger.info(f"Result {i}: video={r['video_id']}, caption count={r['num_references']}")
            if r['num_references'] <= 4:
                logger.warning(f"  !! Video {r['video_id']} only has {r['num_references']} captions!")
    
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Evaluation results saved to {args.output_file}")
    logger.info("Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 