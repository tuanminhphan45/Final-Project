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
from lavis.common.utils import is_url


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
        timesformer_weight_path=None,
        timesformer_model_type="alpro",
    ):
        super().__init__()
        
        self.tokenizer = self.init_tokenizer()
        
        
        # Initialize TimeSformer visual encoder using parent class method
        self.visual_encoder, self.ln_vision = super().init_vision_encoder(
            model_name=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpointing,
            precision=vit_precision,
            num_frames=num_frames
        )
        
        # Store the num_frames as an attribute
        self.num_frames = num_frames
        
        logger = logging.getLogger(__name__)
        logger.info("====== KHỞI TẠO BLIP2TIMESFORMER MODEL ======")
        logger.info(f"Tham số: img_size={img_size}, num_frames={num_frames}, num_query_token={num_query_token}")
        
        # Xác nhận rằng classifier đã được xóa
        if hasattr(self.visual_encoder, "model"):
            if hasattr(self.visual_encoder.model, "head") and self.visual_encoder.model.head is None:
                logger.info("✓ TimeSformer: Classifier đã được xóa thành công (head=None)")
            elif not hasattr(self.visual_encoder.model, "head"):
                logger.info("✓ TimeSformer: Classifier không tồn tại trong model")
            else:
                logger.warning("⚠️ CHÚ Ý: TimeSformer vẫn có classifier head!")
                # Xóa classifier một cách rõ ràng
                logger.info("Đang xóa classifier...")
                self.visual_encoder.model.remove_classifier()
                logger.info("✓ Đã xóa classifier")

        # Nếu có đường dẫn cụ thể, ghi đè weights đã load từ ImageNet
        if timesformer_weight_path:
            logger.info(f"Đang load TimeSformer weights từ đường dẫn được chỉ định: {timesformer_weight_path}")
            self.load_timesformer_from_pretrained(
                timesformer_weight_path, 
                model_type=timesformer_model_type
            )

        # Kiểm tra xem TimeSformer đã có weights hay chưa
        if hasattr(self.visual_encoder, "model") and hasattr(self.visual_encoder.model, "pos_embed"):
            logger.info(f"✓ TimeSformer đã có weights: pos_embed shape = {self.visual_encoder.model.pos_embed.shape}")
            if hasattr(self.visual_encoder.model, "time_embed"):
                logger.info(f"✓ TimeSformer đã có temporal weights: time_embed shape = {self.visual_encoder.model.time_embed.shape}")
        else:
            logger.warning("⚠️ TimeSformer weights không được load đúng cách! Kiểm tra lại cấu hình.")

        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logger.info("✓ Đã đóng băng (freeze) TimeSformer encoder")

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
        logger.info("====== KHỞI TẠO BLIP2TIMESFORMER MODEL HOÀN TẤT ======")

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
        use_nucleus_sampling: bool = True,
        num_beams: int = 3,
        max_length: int = 50,
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
        
        # Lấy đường dẫn và loại model cho TimeSformer
        timesformer_weight_path = None
        timesformer_model_type = cfg.get("timesformer_model_type", "alpro")
        
        if hasattr(cfg, "timesformer_pretrained") and cfg.timesformer_pretrained:
            if is_url(cfg.timesformer_pretrained) or os.path.isfile(cfg.timesformer_pretrained):
                timesformer_weight_path = cfg.timesformer_pretrained

        # Lấy đường dẫn cho QFormer weights (thêm mới)
        qformer_weight_path = None
        if hasattr(cfg, "qformer_pretrained") and cfg.qformer_pretrained:
            if is_url(cfg.qformer_pretrained) or os.path.isfile(cfg.qformer_pretrained):
                qformer_weight_path = cfg.qformer_pretrained
        
        logger = logging.getLogger(__name__)

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
            timesformer_weight_path=timesformer_weight_path,
            timesformer_model_type=timesformer_model_type,
        )
        
        # TimeSformer weights đã được load trong __init__ nếu timesformer_weight_path được cung cấp
        
        # Load QFormer weights nếu có đường dẫn (thêm mới)
        if qformer_weight_path:
            logger.info(f"Đang load QFormer weights từ {qformer_weight_path}")
            msg = model.load_from_pretrained(qformer_weight_path)
            logger.info(f"Load QFormer weights thành công, missing keys: {len(msg.missing_keys)}")
        
        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)

    def load_timesformer_from_pretrained(self, url_or_filename, model_type="alpro"):
        """
        Load pretrained weights cho TimeSformer với hỗ trợ nhiều loại model khác nhau
        Args:
            url_or_filename: Đường dẫn hoặc URL đến file checkpoint
            model_type: Loại model pretrained ('alpro', 'alpro_qa', 'timesformer', hoặc 'custom')
        """
        import os
        from lavis.common.dist_utils import download_cached_file
        from lavis.common.utils import is_url
        
        logger = logging.getLogger(__name__)
        logger.info("---------------------------------------------------")
        logger.info(f"LOADING TIMESFORMER WEIGHTS")
        logger.info("---------------------------------------------------")
        logger.info(f"Source: {url_or_filename}")
        logger.info(f"Model type: {model_type}")
        
        # Load checkpoint
        if is_url(url_or_filename):
            logger.info(f"Downloading weights từ URL...")
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
            logger.info(f"✓ Đã download xong weights từ URL")
        elif os.path.isfile(url_or_filename):
            logger.info(f"Loading weights từ file local...")
            checkpoint = torch.load(url_or_filename, map_location="cpu")
            logger.info(f"✓ Đã load xong weights từ file local")
        else:
            raise RuntimeError(f"✗ LỖI: Đường dẫn không hợp lệ: {url_or_filename}")
        
        # Lấy state_dict
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
            logger.info(f"✓ Phát hiện state_dict trong key 'model'")
        elif "module" in checkpoint:  # Alpro models thường lưu với tiền tố "module"
            state_dict = checkpoint["module"]
            logger.info(f"✓ Phát hiện state_dict trong key 'module'")
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            logger.info(f"✓ Phát hiện state_dict trong key 'state_dict'")
        else:
            state_dict = checkpoint
            logger.info(f"✓ Sử dụng checkpoint trực tiếp làm state_dict")
        
        # Thông số cần thiết
        num_patches = (self.visual_encoder.img_size // self.visual_encoder.patch_size) ** 2
        num_frames = self.visual_encoder.num_frames
        logger.info(f"Cấu hình TimeSformer: {num_patches} patches, {num_frames} frames")
        
        # Xử lý cấu trúc key dựa trên loại model
        visual_encoder_keys = []
        
        # Kiểm tra cấu trúc key
        has_visual_encoder_prefix = any(k.startswith("visual_encoder.") for k in state_dict.keys())
        has_model_prefix = any(k.startswith("model.") for k in state_dict.keys())
        has_visual_model_prefix = any(k.startswith("visual.model.") for k in state_dict.keys())
        has_direct_keys = any(k in ["pos_embed", "cls_token", "time_embed"] for k in state_dict.keys())
        
        # Chuyển đổi cấu trúc key dựa trên loại model
        new_state_dict = {}
        
        # Trường hợp 1: Mô hình Alpro với "visual.model."
        if model_type == "alpro" or model_type == "alpro_qa" or has_visual_model_prefix:
            logger.info("Đang xử lý weights từ mô hình Alpro")
            prefix = "visual.model." if has_visual_model_prefix else ""
            for k, v in state_dict.items():
                if prefix and k.startswith(prefix):
                    new_k = "visual_encoder.model." + k[len(prefix):]
                    new_state_dict[new_k] = v
                    visual_encoder_keys.append(new_k)
                elif k.startswith("model."):
                    new_k = "visual_encoder." + k
                    new_state_dict[new_k] = v
                    visual_encoder_keys.append(new_k)
                elif not has_visual_encoder_prefix and not prefix and not k.startswith("model."):
                    # Direct TimeSformer weights
                    new_k = "visual_encoder.model." + k
                    new_state_dict[new_k] = v
                    visual_encoder_keys.append(new_k)
                elif k.startswith("visual_encoder."):
                    new_state_dict[k] = v
                    visual_encoder_keys.append(k)
        
        # Trường hợp 2: TimeSformer gốc hoặc pretrained model với key trực tiếp
        elif model_type == "timesformer" or has_direct_keys:
            logger.info("Đang xử lý weights từ TimeSformer gốc")
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_k = "visual_encoder." + k
                    new_state_dict[new_k] = v
                    visual_encoder_keys.append(new_k)
                elif not has_visual_encoder_prefix:
                    new_k = "visual_encoder.model." + k
                    new_state_dict[new_k] = v
                    visual_encoder_keys.append(new_k)
                else:
                    new_state_dict[k] = v
                    if k.startswith("visual_encoder."):
                        visual_encoder_keys.append(k)
        
        # Trường hợp 3: Đã có cấu trúc đúng với "visual_encoder."
        elif has_visual_encoder_prefix:
            logger.info("Đã có cấu trúc key đúng với 'visual_encoder.'")
            new_state_dict = state_dict
            visual_encoder_keys = [k for k in state_dict.keys() if k.startswith("visual_encoder.")]
        
        # Trường hợp 4: Custom format - thử tìm cấu trúc
        else:
            logger.info("Tự động phát hiện cấu trúc key...")
            # Tìm các key liên quan đến embedding
            pos_embed_keys = [k for k in state_dict.keys() if "pos_embed" in k]
            if pos_embed_keys:
                logger.info(f"Tìm thấy key có chứa 'pos_embed': {pos_embed_keys[0]}")
                src_key = pos_embed_keys[0]
                prefix = src_key.split("pos_embed")[0]
                
                for k, v in state_dict.items():
                    if k.startswith(prefix):
                        new_k = "visual_encoder.model." + k[len(prefix):]
                        new_state_dict[new_k] = v
                        visual_encoder_keys.append(new_k)
            else:
                logger.warning("Không thể tự động phát hiện cấu trúc key, sử dụng nguyên bản")
                new_state_dict = state_dict
        
        # Kiểm tra và thông báo số lượng key đã tìm thấy
        logger.info(f"Đã tìm thấy {len(visual_encoder_keys)} key cho visual_encoder")
        
        # Xử lý spatial embedding nếu cần
        spatial_embed_key = "visual_encoder.model.pos_embed"
        if spatial_embed_key in new_state_dict and num_patches + 1 != new_state_dict[spatial_embed_key].size(1):
            logger.info(f"Điều chỉnh spatial embedding từ {new_state_dict[spatial_embed_key].size(1)} thành {num_patches + 1}")
            
            pos_embed = new_state_dict[spatial_embed_key]
            cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
            
            new_pos_embed = F.interpolate(other_pos_embed, size=(num_patches), mode="nearest")
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            
            new_state_dict[spatial_embed_key] = new_pos_embed
            
        # Xử lý temporal embedding nếu cần
        temporal_embed_key = "visual_encoder.model.time_embed"
        if temporal_embed_key in new_state_dict and num_frames != new_state_dict[temporal_embed_key].size(1):
            original_frames = new_state_dict[temporal_embed_key].size(1)
            logger.info(f"Điều chỉnh temporal embedding từ {original_frames} thành {num_frames}")
            
            time_embed = new_state_dict[temporal_embed_key].transpose(1, 2)
            new_time_embed = F.interpolate(time_embed, size=(num_frames), mode="nearest")
            new_state_dict[temporal_embed_key] = new_time_embed.transpose(1, 2)
        
        # Khởi tạo temporal attention từ spatial attention nếu cần
        attention_type = "divided_space_time"
        if attention_type == "divided_space_time":
            logger.info("Khởi tạo temporal attention từ spatial attention")
            latest_state_dict = new_state_dict.copy()
            
            for key in new_state_dict:
                if "visual_encoder.model.blocks" in key and "attn" in key and not "temporal_attn" in key:
                    new_key = key.replace("attn", "temporal_attn")
                    if new_key not in new_state_dict:
                        latest_state_dict[new_key] = new_state_dict[key]
                    
                if "visual_encoder.model.blocks" in key and "norm1" in key and not "temporal_norm1" in key:
                    new_key = key.replace("norm1", "temporal_norm1")
                    if new_key not in new_state_dict:
                        latest_state_dict[new_key] = new_state_dict[key]
        
            new_state_dict = latest_state_dict
        
        # Load state_dict vào mô hình
        logger.info("Đang áp dụng weights vào TimeSformer...")
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        
        logger.info("---------------------------------------------------")
        logger.info(f"KẾT QUẢ LOAD TIMESFORMER WEIGHTS:")
        logger.info(f"✓ Số tham số đã load: {len(new_state_dict)} keys")
        logger.info(f"✓ Missing keys: {len(missing)}")
        logger.info(f"✓ Unexpected keys: {len(unexpected)}")
        logger.info("---------------------------------------------------")
        
        return missing, unexpected

    def load_from_pretrained(self, url_or_filename, **kwargs):
        """
        Load pretrained weights vào mô hình BLIP2 TimeSformer
        Lưu ý: Weights cho TimeSformer nên được load thông qua __init__ hoặc load_timesformer_from_pretrained()
        """
        logger = logging.getLogger(__name__)
        logger.info("===== LOAD PRETRAINED BLIP2 WEIGHTS =====")
        
        # Không tự động load TimeSformer weights nữa (đã được xử lý trong __init__)
        # Chỉ truyền vào kwargs["timesformer_url"] nếu muốn ghi đè weights hiện tại
        if "timesformer_url" in kwargs and kwargs["timesformer_url"]:
            logger.info(f"Đang load TimeSformer weights từ {kwargs['timesformer_url']}")
            self.load_timesformer_from_pretrained(kwargs["timesformer_url"])
        
        if is_url(url_or_filename):
            logger.info(f"Đang download checkpoint từ URL: {url_or_filename}")
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
            logger.info("✓ Đã tải checkpoint từ URL thành công")
        elif os.path.isfile(url_or_filename):
            logger.info(f"Đang load checkpoint từ file: {url_or_filename}")
            checkpoint = torch.load(url_or_filename, map_location="cpu")
            logger.info("✓ Đã tải checkpoint từ file thành công")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
            logger.info("Đã tìm thấy state_dict trong key 'model'")
        else:
            state_dict = checkpoint
            logger.info("Sử dụng checkpoint trực tiếp làm state_dict")
            
        # Lọc bỏ các weights của visual_encoder để tránh ghi đè lên TimeSformer đã được load
        filtered_state_dict = {}
        qformer_keys = 0
        other_keys = 0
        
        for key, value in state_dict.items():
            if not key.startswith("visual_encoder"):
                filtered_state_dict[key] = value
                if key.startswith("Qformer.") or key.startswith("query_tokens"):
                    qformer_keys += 1
                else:
                    other_keys += 1
        
        state_dict = filtered_state_dict
        logger.info(f"Đã lọc state_dict: {qformer_keys} QFormer keys, {other_keys} keys khác")
        
        # Kiểm tra xem có weights cho QFormer hay không
        has_qformer_weights = any(k.startswith("Qformer.") for k in state_dict.keys())
        if has_qformer_weights:
            logger.info("✓ Checkpoint có chứa weights cho QFormer")
        else:
            logger.warning("⚠️ Checkpoint KHÔNG chứa weights cho QFormer!")
        
        # Kiểm tra xem có query_tokens hay không
        has_query_tokens = any(k.startswith("query_tokens") for k in state_dict.keys())
        if has_query_tokens:
            logger.info("✓ Checkpoint có chứa weights cho query_tokens")
        else:
            logger.warning("⚠️ Checkpoint KHÔNG chứa weights cho query_tokens!")
        
        # Thêm thông tin về QFormer weights trước khi load
        if hasattr(self, "Qformer") and hasattr(self.Qformer, "bert") and hasattr(self.Qformer.bert, "encoder"):
            sample_layer = self.Qformer.bert.encoder.layer[0]
            if hasattr(sample_layer, "attention"):
                before_weights = sample_layer.attention.self.query.weight.mean().item()
                logger.info(f"QFormer weights trước khi load: mean={before_weights:.4f}")
        
        msg = self.load_state_dict(state_dict, strict=False)
        
        # Kiểm tra QFormer weights sau khi load
        if hasattr(self, "Qformer") and hasattr(self.Qformer, "bert") and hasattr(self.Qformer.bert, "encoder"):
            sample_layer = self.Qformer.bert.encoder.layer[0]
            if hasattr(sample_layer, "attention"):
                after_weights = sample_layer.attention.self.query.weight.mean().item()
                logger.info(f"QFormer weights sau khi load: mean={after_weights:.4f}")
                
                if abs(before_weights - after_weights) > 1e-4:
                    logger.info("✓ THÀNH CÔNG: QFormer weights đã thay đổi sau khi load")
                else:
                    logger.warning("⚠️ QFormer weights KHÔNG thay đổi sau khi load!")
        
        logger.info("Missing keys {}".format(len(msg.missing_keys)))
        if msg.missing_keys:
            logger.info(f"Một số missing keys: {msg.missing_keys[:5]}...")
        
        if msg.unexpected_keys:
            logger.info(f"Unexpected keys: {len(msg.unexpected_keys)}")
            if msg.unexpected_keys:
                logger.info(f"Một số unexpected keys: {msg.unexpected_keys[:5]}...")
        
        logger.info("✓ Đã load checkpoint thành công")
        logger.info("===== KẾT THÚC LOAD PRETRAINED BLIP2 WEIGHTS =====")
        
        return msg
