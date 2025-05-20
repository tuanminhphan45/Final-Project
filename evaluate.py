"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import random
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.utils import now, is_url

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument(
        "--timesformer-checkpoint", 
        help="path to TimeSformer weights (để đảm bảo model luôn load đúng TimeSformer weights)"
    )
    parser.add_argument(
        "--qformer-checkpoint", 
        help="path to QFormer weights (sử dụng khi model checkpoint không chứa weights QFormer)"
    )
    parser.add_argument(
        "--resume-checkpoint", 
        help="path to checkpoint to reload (sẽ load cả TimeSformer và QFormer weights nếu có)"
    )
    parser.add_argument(
        "--skip-reload", 
        action="store_true",
        help="bỏ qua việc load best checkpoint (sử dụng model đã load)"
    )
    
    args = parser.parse_args()

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    args = parse_args()
    cfg = Config(args)

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)
    
    # Bổ sung load weights khi evaluate
    logger = setup_logger()
    logger.info("===== CHUẨN BỊ MODEL CHO EVALUATION =====")
    
    # Load TimeSformer weights nếu được chỉ định
    if args.timesformer_checkpoint:
        if hasattr(model, "load_timesformer_from_pretrained"):
            if os.path.isfile(args.timesformer_checkpoint) or is_url(args.timesformer_checkpoint):
                logger.info(f"Loading TimeSformer weights from {args.timesformer_checkpoint}")
                model.load_timesformer_from_pretrained(args.timesformer_checkpoint)
            else:
                logger.warning(f"⚠️ TimeSformer checkpoint không tồn tại: {args.timesformer_checkpoint}")
        else:
            logger.warning("⚠️ Model không hỗ trợ phương thức load_timesformer_from_pretrained")
            
    # Load QFormer weights nếu được chỉ định
    if args.qformer_checkpoint:
        if os.path.isfile(args.qformer_checkpoint):
            logger.info(f"Loading QFormer checkpoint from {args.qformer_checkpoint}")
            state_dict = torch.load(args.qformer_checkpoint, map_location="cpu")
            if "model" in state_dict:
                # Lọc chỉ lấy các weights của QFormer và query_tokens
                qformer_weights = {k: v for k, v in state_dict["model"].items() 
                                  if k.startswith("Qformer.") or k.startswith("query_tokens")}
                model.load_state_dict(qformer_weights, strict=False)
                logger.info(f"✓ Đã load {len(qformer_weights)} QFormer weights")
            else:
                logger.warning("⚠️ Checkpoint không chứa key 'model'")
        else:
            logger.warning(f"⚠️ QFormer checkpoint không tồn tại: {args.qformer_checkpoint}")
    
    # Load checkpoint đầy đủ nếu được chỉ định
    if args.resume_checkpoint:
        if os.path.isfile(args.resume_checkpoint):
            logger.info(f"Loading model checkpoint from {args.resume_checkpoint}")
            checkpoint = torch.load(args.resume_checkpoint, map_location="cpu")
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"], strict=False)
                logger.info(f"✓ Đã load model checkpoint")
            else:
                logger.warning("⚠️ Checkpoint không chứa key 'model'")
        else:
            logger.warning(f"⚠️ Checkpoint không tồn tại: {args.resume_checkpoint}")

    runner = RunnerBase(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.evaluate(skip_reload=args.skip_reload)


if __name__ == "__main__":
    main()
