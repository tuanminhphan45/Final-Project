"""
Copyright (c) 2023, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, compute_sim_matrix, disabled_train
from lavis.models.timesformer.vit import TimeSformer
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
import logging
import os
import gc
from lavis.common.dist_utils import download_cached_file
import traceback


@registry.register_model("blip2_timesformer_memory_optimized")
class Blip2TimeSformerMemoryOptimized(Blip2Base):
    """
    Phiên bản tối ưu bộ nhớ của BLIP2TimeSformer.
    """
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_timesformer.yaml",
    }

    def __init__(
        self,
        vit_model="timesformer",
        img_size=224,
        num_frames=16,
        drop_path_rate=0.1,
        use_grad_checkpointing=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,  # Giảm số query token để tiết kiệm bộ nhớ
        cross_attention_freq=2,
        embed_dim=768,
        max_txt_len=32,
        
    ):
        super().__init__()
        
        self.tokenizer = self.init_tokenizer()
        
        # Initialize TimeSformer visual encoder
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, num_frames, drop_path_rate, use_grad_checkpointing, "fp16"
        )
        
        # Store the num_frames as an attribute
        self.num_frames = num_frames

        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
        
        # Initialize Q-Former with correct embed_dim
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.model.embed_dim, cross_attention_freq
        )
        
        #Resize token embeddings để phù hợp với số lượng token trong tokenizer
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        
        # Khởi tạo query parameters từ state dict
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])
        
        # Projection layers
        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
        
        # Temperature parameter
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        
        self.max_txt_len = max_txt_len
    
    def init_vision_encoder(
        self,
        vit_model="timesformer",
        img_size=224,
        num_frames=8,
        drop_path_rate=0.1,
        use_grad_checkpointing=False,
        vit_precision="fp16",
    ):
        """Initialize TimeSformer visual encoder"""
        logger = logging.getLogger(__name__)
        logger.info(f"Initializing TimeSformer with img_size={img_size}, patch_size=16, num_frames={num_frames}")
        
        # Initialize TimeSformer
        visual_encoder = TimeSformer(
            image_size=img_size,
            patch_size=16,
            n_frms=num_frames,
            attn_drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            drop_rate=0,
            use_grad_ckpt=use_grad_checkpointing,
            ckpt_layer=0,
            remove_classifier=True,
        )
        # Get embedding dimension from TimeSformer's model
        embed_dim = visual_encoder.model.embed_dim
        ln_vision = nn.LayerNorm(embed_dim)
        
        return visual_encoder, ln_vision

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling: bool = False,
        num_beams: int = 2,  # Giảm số beam để tiết kiệm bộ nhớ
        max_length: int = 30,
        min_length: int = 10,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        batch_size_limit: int = None,  # Thêm tham số để hạn chế kích thước batch
    ):
        """
        Generate captions cho video input với tối ưu hóa bộ nhớ.
        
        Args:
            samples: Dict hoặc Tensor chứa video
            batch_size_limit: Giới hạn kích thước batch để xử lý theo loạt
        """
        logger = logging.getLogger(__name__)

        # 1) Unpack video
        if not hasattr(samples, "get"):
            video = samples
        else:
            video = samples.get("video", None)
        if video is None:
            logger.error("No video provided for caption generation")
            return ["No video provided"]
        
        # Lấy sample_ids nếu có (để ghép lại kết quả sau)
        if hasattr(samples, "get") and "image_id" in samples:
            sample_ids = samples["image_id"]
        else:
            sample_ids = None
        
        orig_batch_size = video.size(0)
        
        # Xử lý theo batch nhỏ nếu cần để tiết kiệm bộ nhớ
        if batch_size_limit is not None and orig_batch_size > batch_size_limit:
            logger.info(f"Processing in mini-batches of {batch_size_limit} samples")
            all_captions = []
            
            for i in range(0, orig_batch_size, batch_size_limit):
                end_idx = min(i + batch_size_limit, orig_batch_size)
                mini_batch = video[i:end_idx]
                
                mini_samples = {"video": mini_batch}
                if sample_ids is not None:
                    mini_samples["image_id"] = sample_ids[i:end_idx]
                
                # Gọi đệ quy với mini-batch
                mini_captions = self.generate(
                    mini_samples,
                    use_nucleus_sampling=use_nucleus_sampling,
                    num_beams=num_beams,
                    max_length=max_length,
                    min_length=min_length,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    batch_size_limit=None  # Tránh đệ quy vô hạn
                )
                
                all_captions.extend(mini_captions)
                
                # Dọn bộ nhớ sau mỗi lần chạy
                torch.cuda.empty_cache()
                gc.collect()
            
            return all_captions
        
        batch_size = video.size(0)

        # 2) Encode video
        with torch.cuda.amp.autocast(enabled=True):  # Sử dụng mixed precision
            video_embeds = self.ln_vision(self.visual_encoder.forward_features(video))

        # Dọn bộ nhớ sau khi encode
        torch.cuda.empty_cache()
        
        video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(video.device)

        # 3) Số beam thực tế có thể cần điều chỉnh dựa trên kích thước batch để tránh OOM
        if batch_size > 4 and num_beams > 1:
            # Giảm số beam khi batch size lớn
            effective_num_beams = max(1, num_beams - 1)
            logger.info(f"Reduced beam size to {effective_num_beams} due to large batch size ({batch_size})")
        else:
            effective_num_beams = num_beams

        # Lượt bỏ tilebeam cho batch lớn nếu cần
        if not use_nucleus_sampling and batch_size > 8:
            # Không nhân bản video_embeds để tiết kiệm bộ nhớ
            logger.info("Skipping beam tiling to save memory")
            model_kwargs = {
                "encoder_hidden_states": video_embeds,
                "encoder_attention_mask": video_atts,
            }
        else:
            if not use_nucleus_sampling:
                video_embeds = video_embeds.repeat_interleave(effective_num_beams, dim=0)
                video_atts = video_atts.repeat_interleave(effective_num_beams, dim=0)
            
            model_kwargs = {
                "encoder_hidden_states": video_embeds,
                "encoder_attention_mask": video_atts,
            }
        
        # Chuẩn bị input_ids
        input_ids = torch.full(
            (video_embeds.size(0), 1),
            self.tokenizer.bos_token_id,
            dtype=torch.long,
            device=video.device,
        )

        # Chuẩn bị query tokens
        query_tokens = self.query_tokens.expand(video_embeds.size(0), -1, -1)
        
        # Giải phóng bộ nhớ không cần thiết
        torch.cuda.empty_cache()
        
        try:
            # 4) Generate với max_memory_usage
            with torch.cuda.amp.autocast(enabled=True):  # Sử dụng mixed precision
                outputs = self.Qformer.generate(
                    input_ids=input_ids,
                    query_embeds=query_tokens,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=effective_num_beams,
                    do_sample=use_nucleus_sampling,
                    top_p=top_p,
                    eos_token_id=self.tokenizer.sep_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **model_kwargs
                )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                # Nếu vẫn OOM, thử lại với tham số tối thiểu
                logger.warning("OOM during generation. Trying with minimal parameters...")
                torch.cuda.empty_cache()
                gc.collect()
                
                # Giảm beam size và length tối đa
                return self.generate(
                    samples,
                    use_nucleus_sampling=True,  # Switch to sampling
                    num_beams=1,
                    max_length=20,
                    min_length=5,
                    top_p=0.9,
                    repetition_penalty=1.0,
                    batch_size_limit=1  # Process one by one
                )
            else:
                raise e
                
        # 5) Decode captions
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Dọn dẹp bộ nhớ
        torch.cuda.empty_cache()
        gc.collect()
        
        return captions

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "timesformer")
        img_size = cfg.get("image_size", 224)
        num_frames = cfg.get("num_frames", 8)
        drop_path_rate = cfg.get("drop_path_rate", 0.1)
        use_grad_checkpointing = cfg.get("use_grad_checkpoint", False)
        vit_precision = "fp16"
        freeze_vit = cfg.get("freeze_vit", True)
        num_query_token = cfg.get("num_query_token", 32)  # Mặc định ít token hơn
        cross_attention_freq = cfg.get("cross_attention_freq", 2)
        embed_dim = cfg.get("embed_dim", 768)
        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            num_frames=num_frames,
            drop_path_rate=drop_path_rate,
            use_grad_checkpointing=use_grad_checkpointing,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            embed_dim=embed_dim,
            max_txt_len=max_txt_len,
        )

        return model 