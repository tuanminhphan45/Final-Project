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


@registry.register_model("blip2_timesformer")
class Blip2TimeSformer(Blip2Base):
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
        num_query_token=32,
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

    def forward(self, samples):
        """
        Args:
            samples (dict): A dictionary containing:
                - video (torch.Tensor): Shape [B, T, C, H, W]
                - text_input (list): List of text strings
        """
        video = samples["video"]  # [B, T, C, H, W]
        text = samples["text_input"]

        if video.dim() != 5:
          raise ValueError(f"Expect 5‑D video tensor, got {video.shape}")

        if video.shape[1] == 3:
          pass

        elif video.shape[2] == 3:
          video = video.permute(0, 2, 1, 3, 4).contiguous()

        else:
          raise ValueError(f"Unexpected layout: {video.shape}")

        # Forward video qua TimeSformer
        video_embeds = self.visual_encoder.forward_features(video)  # [B, 1+T×N, D]
        video_embeds = self.ln_vision(video_embeds)
        video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(video.device)
        
        logger = logging.getLogger(__name__)
        # Prepare query tokens
        query_tokens = self.query_tokens.expand(video_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=video_embeds,
            encoder_attention_mask=video_atts,
            use_cache=True,
            return_dict=True,
        )
        # Project features to embedding space
        video_feats = F.normalize(self.vision_proj(query_output.last_hidden_state), dim=-1)

        
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
        video_feats_all = concat_all_gather(
            video_feats
        )
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(
            video_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), video_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

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
        #attention_mask = text_tokens.attention_mask
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
        Args:
            samples (dict or Tensor): If dict, expects {"video": Tensor[B,C,T,H,W]};
                                    otherwise interpreted directly as the video tensor.
            use_nucleus_sampling: if True, samples with top-p; else uses beam search.
            num_beams: number of beams (only used when not sampling).
            max_length/min_length: decoding length bounds.
            top_p: nucleus sampling probability mass.
            repetition_penalty: penalize repeated tokens.
            length_penalty: beam-search length penalty.
        Returns:
            List[str]: decoded captions of length B.
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

        # 2) Encode video → (B, Q, D)
        video_embeds = self.ln_vision(self.visual_encoder.forward_features(video))

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
        batch_beam = video_embeds.size(0)
        input_ids = torch.full(
            (batch_beam, 1),
            self.tokenizer.bos_token_id,
            dtype=torch.long,
            device=video.device,
        )

        # 6) Tile Q-Former's fixed query tokens to match batch
        # query_tokens = self.query_tokens.expand(video_embeds.size(0), -1, -1)
        query_tokens = self.query_tokens.expand(batch_beam, -1, -1)
    
        # 7) Call HF generate
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
    def forward_video(self, video):
        video_embeds = self.ln_vision(self.visual_encoder.forward_features(video))
        video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(
            video.device
        )

        query_tokens = self.query_tokens.expand(video_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=video_embeds,
            encoder_attention_mask=video_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, video_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, video_inputs, text_ids, text_atts):
        video_atts = torch.ones(video_inputs.size()[:-1], dtype=torch.long).to(
            video_inputs.device
        )
        query_tokens = self.query_tokens.expand(video_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            video_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=video_inputs,
            encoder_attention_mask=video_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit
        

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - video (torch.Tensor): A tensor of shape (B, C, T, H, W) or (B, T, C, H, W)
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "video".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
        """
        video = samples.get("video")
        caption = samples.get("text_input")

        # assert mode is one of "video", "text", "multimodal"
        assert mode in [
            "video",
            "text",
            "multimodal",
        ], "mode must be one of 'video', 'text', 'multimodal'"

        # initalize output
        video_embeds, text_embeds, multimodal_embeds = None, None, None
        video_features, text_features = None, None

        if mode == "video":
            assert (
                video is not None
            ), "video is not provided for mode 'video' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                video_embeds_frozen = self.visual_encoder.forward_features(video)
            video_embeds_frozen = video_embeds_frozen.float()
            video_atts = torch.ones(
                video_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                video_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=video_embeds_frozen,
                encoder_attention_mask=video_atts,
                return_dict=True,
            )
            video_embeds = query_output.last_hidden_state
            video_features = F.normalize(self.vision_proj(video_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                video_embeds_frozen = self.ln_vision(self.visual_encoder.forward_features(video))
            video_embeds_frozen = video_embeds_frozen.float()
            video_atts = torch.ones(
                video_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                video_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=video_embeds_frozen,
                encoder_attention_mask=video_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=video_embeds,
            image_embeds_proj=video_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "timesformer")
        img_size = cfg.get("image_size", 224)
        num_frames = cfg.get("num_frames", 16)
        drop_path_rate = cfg.get("drop_path_rate", 0.1)
        use_grad_checkpointing = cfg.get("use_grad_checkpointing", False)
        vit_precision = "fp16"
        freeze_vit = cfg.get("freeze_vit", True)
        num_query_token = cfg.get("num_query_token", 128)
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
        
        # Load pretrained weights cho TimeSformer từ config file
        if hasattr(cfg, "timesformer_pretrained") and cfg.timesformer_pretrained:
            timesformer_url = cfg.timesformer_pretrained
            if is_url(timesformer_url) or os.path.isfile(timesformer_url):
                model.load_timesformer_from_pretrained(timesformer_url)
            else:
                logging.warning(f"TimeSformer pretrained path {timesformer_url} không hợp lệ")
        
        # Load pretrained weights cho Q-former từ config file
        if hasattr(cfg, "pretrained_qformer_path") and cfg.pretrained_qformer_path:
            pretrained_qformer_path = cfg.pretrained_qformer_path
            if is_url(pretrained_qformer_path) or os.path.isfile(pretrained_qformer_path):
                model.load_from_pretrained(pretrained_qformer_path, 
                                           timesformer_url=cfg.get("timesformer_pretrained", None))
            else:
                logging.warning(f"Pretrained Q-former path {pretrained_qformer_path} không hợp lệ")
        
        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)

    def load_timesformer_from_pretrained(self, url_or_filename):
        """
        Load pretrained weights cho TimeSformer
        """
        from lavis.models.timesformer.helpers import load_state_dict
        import os
        
        logger = logging.getLogger(__name__)
        logger.info(f"Loading TimeSformer pretrained weights từ {url_or_filename}")
        
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = load_state_dict(cached_file)
        elif os.path.isfile(url_or_filename):
            checkpoint = load_state_dict(url_or_filename)
        else:
            raise RuntimeError("checkpoint url or path is invalid")
            
        # Tính toán số patches và frames
        num_patches = (self.visual_encoder.img_size // self.visual_encoder.patch_size) ** 2
        num_frames = self.visual_encoder.num_frames
        
        # Xử lý embedding cho TimeSformer
        if "pos_embed" in checkpoint and num_patches + 1 != checkpoint["pos_embed"].size(1):
            logger.info(f"Điều chỉnh spatial embedding từ {checkpoint['pos_embed'].size(1)} thành {num_patches + 1}")
            pos_embed = checkpoint["pos_embed"]
            cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
            new_pos_embed = F.interpolate(other_pos_embed, size=(num_patches), mode="nearest")
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            checkpoint["pos_embed"] = new_pos_embed
        
        if "time_embed" in checkpoint and num_frames != checkpoint["time_embed"].size(1):
            logger.info(f"Điều chỉnh temporal embedding từ {checkpoint['time_embed'].size(1)} thành {num_frames}")
            time_embed = checkpoint["time_embed"].transpose(1, 2)
            new_time_embed = F.interpolate(time_embed, size=(num_frames), mode="nearest")
            checkpoint["time_embed"] = new_time_embed.transpose(1, 2)
        
        # Khởi tạo temporal attention nếu cần
        attention_type = "divided_space_time"
        if attention_type == "divided_space_time":
            new_state_dict = checkpoint.copy()
            for key in checkpoint:
                if "blocks" in key and "attn" in key:
                    new_key = key.replace("attn", "temporal_attn")
                    if not new_key in checkpoint:
                        new_state_dict[new_key] = checkpoint[key]
                    else:
                        new_state_dict[new_key] = checkpoint[new_key]
                if "blocks" in key and "norm1" in key:
                    new_key = key.replace("norm1", "temporal_norm1")
                    if not new_key in checkpoint:
                        new_state_dict[new_key] = checkpoint[key]
                    else:
                        new_state_dict[new_key] = checkpoint[new_key]
            checkpoint = new_state_dict
        
        # Thêm tiền tố "model." vào tất cả keys
        new_state_dict = {}
        for key, value in checkpoint.items():
            new_state_dict[f"model.{key}"] = value
        
        # Load weights vào mô hình
        missing, unexpected = self.visual_encoder.load_state_dict(new_state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys khi load TimeSformer: {len(missing)} keys")
        if unexpected:
            logger.warning(f"Unexpected keys khi load TimeSformer: {len(unexpected)} keys")
        
        logger.info("TimeSformer pretrained weights loaded successfully")
        
    def load_from_pretrained(self, url_or_filename, **kwargs):
        """
        Load pretrained weights vào mô hình BLIP2 TimeSformer
        """
        if "timesformer_url" in kwargs:
            # Load TimeSformer pretrained weights trước
            self.load_timesformer_from_pretrained(kwargs["timesformer_url"])
        
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
            
        # Lọc bỏ các weights của visual_encoder nếu đã load pretrained weights
        if "timesformer_url" in kwargs:
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if not key.startswith("visual_encoder"):
                    filtered_state_dict[key] = value
            state_dict = filtered_state_dict
        
        msg = self.load_state_dict(state_dict, strict=False)
        
        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)
        
        return msg
