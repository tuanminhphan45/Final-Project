"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib
import logging
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

import lavis.common.dist_utils as dist_utils
from lavis.common.dist_utils import download_cached_file
from lavis.common.utils import is_url
from lavis.common.logger import MetricLogger
from lavis.models.base_model import BaseModel
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from lavis.models.eva_vit import create_eva_vit_g
from lavis.models.clip_vit import create_clip_vit_L
from lavis.models.timesformer.vit import TimeSformer
from transformers import BertTokenizer


def create_timesformer(img_size, num_frames, drop_path_rate, use_grad_checkpoint, precision=None):
    """
    Create TimeSformer model for video processing
    """
    visual_encoder = TimeSformer(
        image_size=img_size,
        patch_size=16,
        n_frms=num_frames,
        attn_drop_rate=0.0,
        drop_path_rate=drop_path_rate,
        drop_rate=0,
        use_grad_ckpt=use_grad_checkpoint,
        ckpt_layer=0,
        remove_classifier=True,
    )
    
    # Load weights từ Kinetics pretrained
    logging.info("Đang load TimeSformer weights từ Kinetics pretrained")
    try:
        # Đường dẫn đến pretrained Kinetics weights
        pretrained_path = "/storage/student10/vidcaption/LAVIS/TimeSformer_divST_8x32_224_K400.pyth"
        
        # Gọi phương thức load_checkpoint của TimeSformer
        success = visual_encoder.load_checkpoint(pretrained_path)
        
        if success:
            logging.info("✓ Đã load TimeSformer weights từ Kinetics thành công")
        else:
            logging.warning("⚠️ Lỗi khi load TimeSformer weights")
    except Exception as e:
        logging.warning(f"⚠️ Không thể load TimeSformer weights từ Kinetics: {str(e)}")
        logging.warning("Model sẽ được khởi tạo với trọng số ngẫu nhiên")
    
    return visual_encoder

class Blip2Base(BaseModel):
    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def init_vision_encoder(
        self, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision, num_frames=None
    ):
        assert model_name in [
            "eva_clip_g",
            "eva2_clip_L",
            "clip_L",
            "timesformer",
        ], "vit model must be eva_clip_g, eva2_clip_L, clip_L, or timesformer"
        
        if model_name == "eva_clip_g":
            visual_encoder = create_eva_vit_g(
                img_size, drop_path_rate, use_grad_checkpoint, precision
            )
#         elif model_name == "eva2_clip_L":
#             visual_encoder = create_eva2_vit_L(
#                 img_size, drop_path_rate, use_grad_checkpoint, precision
#             )
        elif model_name == "clip_L":
            visual_encoder = create_clip_vit_L(img_size, use_grad_checkpoint, precision)
        elif model_name == "timesformer":
            assert num_frames is not None, "num_frames must be provided for timesformer"
            visual_encoder = create_timesformer(
                img_size, num_frames, drop_path_rate, use_grad_checkpoint, precision
            )
            
        # Xác định số chiều đầu ra của visual encoder
        if model_name == "timesformer":
            embed_dim = visual_encoder.model.embed_dim
        else:
            embed_dim = visual_encoder.num_features
            
        ln_vision = LayerNorm(embed_dim)
        self.vit_name = model_name
        return visual_encoder, ln_vision

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

    def get_optimizer_params(self, weight_decay, lr_scale=1):
        if self.visual_encoder is None:
            return []

        # Xử lý đặc biệt cho TimeSformer
        if hasattr(self.visual_encoder, 'model'):
            # TimeSformer stores layers in model.blocks
            vit_num_layers = len(self.visual_encoder.model.blocks)
        else:
            # Fallback cho các encoder khác
            vit_num_layers = self.visual_encoder.get_num_layer() 

        optimizer_config = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if (
                        "visual_encoder" not in n
                        and "prompt_learner" not in n
                        and p.requires_grad
                    )
                ],
                "weight_decay": weight_decay,
                "lr_scale": lr_scale,
            },
        ]

        # Thêm các tham số của visual encoder với learning rate scales khác nhau
        # Thay vì kiểm tra self.cfg.get("freeze_vit"), kiểm tra freeze_vit attribute trực tiếp
        freeze_vit = getattr(self, "freeze_vit", False)
        if not freeze_vit:
            optimizer_config.extend([
                {
                    "params": [
                        p
                        for n, p in self.visual_encoder.named_parameters()
                        if "blocks" in n and p.requires_grad
                    ],
                    "weight_decay": weight_decay,
                    "lr_scale": lr_scale * 0.1,  # Giảm learning rate cho visual encoder
                },
            ])

        return optimizer_config


    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


def compute_sim_matrix(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(model.device)
        text_feat = model.forward_text(text_input)
        text_embed = F.normalize(model.text_proj(text_feat))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    vit_feats = []
    image_embeds = []
    for samples in data_loader:
        image = samples["image"]

        image = image.to(model.device)
        image_feat, vit_feat = model.forward_image(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)

        vit_feats.append(vit_feat.cpu())
        image_embeds.append(image_embed)

    vit_feats = torch.cat(vit_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = []
    for image_embed in image_embeds:
        sim_q2t = image_embed @ text_embeds.t()
        sim_i2t, _ = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = torch.stack(sims_matrix, dim=0)

    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[topk_idx],
            text_atts=text_atts[topk_idx],
        ).float()
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[start + i].repeat(k_test, 1),
            text_atts=text_atts[start + i].repeat(k_test, 1),
        ).float()
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
