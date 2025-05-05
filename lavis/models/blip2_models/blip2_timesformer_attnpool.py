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
from lavis.common.dist_utils import download_cached_file
import traceback


class TemporalAttentionPooling(nn.Module):
    """
    Temporal Attention Pooling module để tổng hợp đặc trưng video qua các frame.
    """
    def __init__(self, dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        
        if num_heads == 1:
            # Single-head attention
            self.attn_fc = nn.Linear(dim, 1)
        else:
            # Multi-head attention
            self.head_dim = dim // num_heads
            assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
            self.attn_proj = nn.Linear(dim, num_heads)
            
        # Thêm layer normalization để ổn định huấn luyện
        self.norm = nn.LayerNorm(dim)
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def forward(self, patch_tokens):
        """
        Args:
            patch_tokens: Tensor với shape [B, T, N, D] 
                B: batch size
                T: số frame
                N: số patch mỗi frame
                D: chiều của embedding
        Returns:
            pooled_patches: Tensor với shape [B, N, D]
        """
        B, T, N, D = patch_tokens.shape
        
        # Chuẩn hóa trước khi tính attention
        patch_tokens = self.norm(patch_tokens)
        
        # Tính frame-level representation bằng cách lấy trung bình qua các patch
        frame_repr = patch_tokens.mean(dim=2)  # [B, T, D]
        
        if self.num_heads == 1:
            # Single-head attention pooling
            scores = self.attn_fc(frame_repr).squeeze(-1)  # [B, T]
            attn_weights = torch.softmax(scores, dim=1)    # [B, T]
            
            # [DEBUG] Log attention weights cho frame đầu tiên trong batch
            if torch.distributed.get_rank() == 0 and torch.rand(1).item() < 0.05:  # log ~5% của các batch
                self.logger.info(f"Temporal Attn Weights: {attn_weights[0].detach().cpu().tolist()}")
            
            # Pooling qua các frame
            attn_weights = attn_weights.unsqueeze(1).unsqueeze(-1)  # [B, 1, T, 1]
            pooled_patches = torch.sum(
                patch_tokens.permute(0, 2, 1, 3) * attn_weights, dim=2
            )  # [B, N, D]
        else:
            # Multi-head attention pooling (advanced)
            scores = self.attn_proj(frame_repr)  # [B, T, num_heads]
            attn_weights = torch.softmax(scores, dim=1)    # [B, T, num_heads]
            
            # Reshape để tính multi-head pooling
            patch_tokens_reshaped = patch_tokens.view(B, T, N, self.num_heads, self.head_dim)
            attn_weights = attn_weights.unsqueeze(2).unsqueeze(-1)  # [B, T, 1, num_heads, 1]
            
            # Pool riêng cho từng head
            pooled = (patch_tokens_reshaped.permute(0, 2, 1, 3, 4) * 
                     attn_weights.permute(0, 2, 1, 3, 4)).sum(dim=2)  # [B, N, num_heads, head_dim]
            pooled_patches = pooled.reshape(B, N, D)  # [B, N, D]
        
        return pooled_patches


@registry.register_model("blip2_timesformer_attnpool")
class Blip2TimeSformerAttnPool(Blip2Base):
    """
    BLIP2 với TimeSformer và Temporal Attention Pooling.
    """
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_timesformer_attnpool.yaml",
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
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=768,
        max_txt_len=32,
        attn_pool_heads=1,  # Số heads cho Temporal Attention Pooling
    ):
        super().__init__()
        
        self.tokenizer = self.init_tokenizer()
        
        # Initialize TimeSformer visual encoder
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, num_frames, drop_path_rate, use_grad_checkpointing, vit_precision
        )
        
        # Store the num_frames as an attribute
        self.num_frames = num_frames
        
        # Tạo Temporal Attention Pooling
        self.temporal_pool = TemporalAttentionPooling(
            dim=self.visual_encoder.model.embed_dim,
            num_heads=attn_pool_heads
        )

        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
        
        # Initialize Q-Former with correct embed_dim
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.model.embed_dim, cross_attention_freq
        )
        
        # Resize token embeddings để phù hợp với số lượng token trong tokenizer
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

    def forward(self, samples):
        """
        Args:
            samples (dict): A dictionary containing:
                - video (torch.Tensor): Shape [B, T, C, H, W] or [B, C, T, H, W]
                - text_input (list): List of text strings
        """
        video = samples["video"]  # [B, T, C, H, W] hoặc [B, C, T, H, W]
        text = samples["text_input"]

        if video.dim() != 5:
          raise ValueError(f"Expect 5‑D video tensor, got {video.shape}")

        # Chuẩn hóa format video: [B, C, T, H, W]
        if video.shape[1] == 3:
          pass  # Already [B, C, T, H, W]
        elif video.shape[2] == 3:
          video = video.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W] -> [B, C, T, H, W]
        else:
          raise ValueError(f"Unexpected layout: {video.shape}")

        # Extract patches from video using TimeSformer
        video_embeds_raw = self.visual_encoder.forward_features(video)  # [B, 1+T×N, D]
        
        # Reshape để áp dụng Temporal Attention Pooling
        B = video_embeds_raw.shape[0]
        D = video_embeds_raw.shape[-1]
        
        # Tách cls token và patch tokens
        cls_token = video_embeds_raw[:, 0:1, :]  # [B, 1, D]
        patch_tokens = video_embeds_raw[:, 1:, :]  # [B, T×N, D]
        
        # Tính toán số patches mỗi frame
        H = W = self.img_size // 16  # ViT sử dụng patch size 16
        N = H * W  # Số patches mỗi frame
        T = self.num_frames  # Số frames
        
        # Reshape patch_tokens để thể hiện rõ cấu trúc temporal
        patch_tokens = patch_tokens.reshape(B, T, N, D)  # [B, T, N, D]
        
        # Áp dụng Temporal Attention Pooling để tổng hợp thông tin từ các frame
        pooled_patches = self.temporal_pool(patch_tokens)  # [B, N, D]
        
        # Combine cls token với pooled patches
        video_embeds = torch.cat([cls_token, pooled_patches], dim=1)  # [B, 1+N, D]
        
        # Apply layer normalization
        video_embeds = self.ln_vision(video_embeds)
        
        # Create attention mask for video embeddings
        video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(video.device)
        
        # Prepare query tokens
        query_tokens = self.query_tokens.expand(video_embeds.shape[0], -1, -1)

        # Forward through Qformer
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=video_embeds,
            encoder_attention_mask=video_atts,
            use_cache=True,
            return_dict=True,
        )
        
        # Project features to embedding space
        video_feats = F.normalize(self.vision_proj(query_output.last_hidden_state), dim=-1)

        # Process text
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(video.device)
        
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )
        
        ###============== Video-text Contrastive ===================###
        video_feats_all = concat_all_gather(video_feats)
        text_feat_all = concat_all_gather(text_feat)

        sim_q2t = torch.matmul(
            video_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), video_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp

        rank = dist.get_rank()
        bs = video.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            video.device
        )

        if "image_id" in samples.keys(): #coco retrieval finetuning
            video_ids = (torch.tensor(samples["image_id"]).view(-1,1)).to(video.device)
            video_ids_all = concat_all_gather(video_ids)
            pos_idx = torch.eq(video_ids, video_ids_all.t()).float()       
            sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)   
            sim_targets = 0.9 * sim_targets + 0.1 * torch.ones_like(sim_targets) / sim_targets.size(1)

            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean()
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()     
            loss_itc = (loss_t2i+loss_i2t)/2
            
        else:
            loss_itc = (
                F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
            ) / 2
        
        ###============== Video-text Matching ===================###
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        video_embeds_world = all_gather_with_grad(video_embeds)
        with torch.no_grad():
            if "image_id" in samples.keys():
                mask = torch.eq(video_ids, video_ids_all.t())
                sim_t2i.masked_fill_(mask, -10000)
                sim_i2t.masked_fill_(mask, -10000)
            else:    
                sim_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
                sim_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)            
                
            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_i2t = F.softmax(sim_i2t, dim=1)

        # select a negative video for each text
        video_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            video_embeds_neg.append(video_embeds_world[neg_idx])
        video_embeds_neg = torch.stack(video_embeds_neg, dim=0)

        # select a negative text for each video
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            video.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        video_embeds_all = torch.cat(
            [video_embeds, video_embeds_neg, video_embeds], dim=0
        )  # pos, neg, pos
        video_atts_all = torch.ones(video_embeds_all.size()[:-1], dtype=torch.long).to(
            video.device
        )

        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=video_embeds_all,
            encoder_attention_mask=video_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(video.device)
        loss_itm = F.cross_entropy(logits, itm_labels)

        ##================= Video Captioning ========================##
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            video.device
        )
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss

        return BlipOutput(
            loss=loss_itc + loss_itm + loss_lm,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm,
        )

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling: bool = False,
        num_beams: int = 3,
        max_length: int = 30,
        min_length: int = 10,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ):
        """
        Generate captions for video input.
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
        
        batch_size = video.size(0)

        # Chuẩn hóa format video: [B, C, T, H, W]
        if video.dim() == 5:
            if video.shape[1] == 3:
                pass  # Already [B, C, T, H, W]
            elif video.shape[2] == 3:
                video = video.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W] -> [B, C, T, H, W]
            else:
                raise ValueError(f"Unexpected layout: {video.shape}")

        # Extract patches from video using TimeSformer
        video_embeds_raw = self.visual_encoder.forward_features(video)  # [B, 1+T×N, D]
        
        # Reshape để áp dụng Temporal Attention Pooling
        B = video_embeds_raw.shape[0]
        D = video_embeds_raw.shape[-1]
        
        # Tách cls token và patch tokens
        cls_token = video_embeds_raw[:, 0:1, :]  # [B, 1, D]
        patch_tokens = video_embeds_raw[:, 1:, :]  # [B, T×N, D]
        
        # Tính toán số patches mỗi frame
        H = W = self.img_size // 16  # ViT sử dụng patch size 16
        N = H * W  # Số patches mỗi frame
        T = self.num_frames  # Số frames
        
        # Reshape patch_tokens để thể hiện rõ cấu trúc temporal
        patch_tokens = patch_tokens.reshape(B, T, N, D)  # [B, T, N, D]
        
        # Áp dụng Temporal Attention Pooling để tổng hợp thông tin từ các frame
        pooled_patches = self.temporal_pool(patch_tokens)  # [B, N, D]
        
        # Combine cls token với pooled patches
        video_embeds = torch.cat([cls_token, pooled_patches], dim=1)  # [B, 1+N, D]
        
        # Apply layer normalization
        video_embeds = self.ln_vision(video_embeds)

        # 3) Tile for beam search if needed
        if not use_nucleus_sampling:
            video_embeds = video_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1

        video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(
            video.device
        )

        model_kwargs = {
            "encoder_hidden_states": video_embeds,
            "encoder_attention_mask": video_atts,
        }
        
        input_ids = torch.full(
            (video_embeds.size(0), 1),
            self.tokenizer.bos_token_id,
            dtype=torch.long,
            device=video.device,
        )

        # Chuẩn bị query tokens
        query_tokens = self.query_tokens.expand(video_embeds.size(0), -1, -1)
    
        # Sinh caption
        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )

        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "timesformer")
        img_size = cfg.get("image_size", 224)
        num_frames = cfg.get("num_frames", 8)
        drop_path_rate = cfg.get("drop_path_rate", 0.1)
        use_grad_checkpointing = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        num_query_token = cfg.get("num_query_token", 32)
        cross_attention_freq = cfg.get("cross_attention_freq", 2)
        embed_dim = cfg.get("embed_dim", 768)
        max_txt_len = cfg.get("max_txt_len", 32)
        attn_pool_heads = cfg.get("attn_pool_heads", 1)

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
            attn_pool_heads=attn_pool_heads,
        )

        return model 