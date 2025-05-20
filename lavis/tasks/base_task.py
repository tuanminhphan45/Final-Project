"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os

import torch
import torch.distributed as dist
from lavis.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample


class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]
            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()

            datasets[name] = dataset

        return datasets

    def train_step(self, model, samples):
        output = model(samples)
        loss_dict = {}
        for k,v in output.items():
            if "loss" in k:
                loss_dict[k] = v
        return output["loss"], loss_dict

    def valid_step(self, model, samples):
        raise NotImplementedError
    
    def before_training(self, model, dataset, **kwargs):
        model.before_training(dataset=dataset, task_type=type(self))

    def before_evaluation(self, model, dataset, **kwargs):
        """
        Thực hiện các công việc cần thiết trước khi đánh giá
        Ví dụ: Đặt model vào chế độ đúng đắn, kiểm tra weights, etc.
        """
        logger = logging.getLogger(__name__)
        
        logger.info("=== CHUẨN BỊ MODEL TRƯỚC KHI ĐÁNH GIÁ ===")
        
        # Kiểm tra model là Blip2TimeSformer hay không
        if hasattr(model, "visual_encoder") and hasattr(model, "Qformer"):
            logger.info("✓ Model là Blip2 với TimeSformer")
            
            # Kiểm tra trạng thái TimeSformer weights
            if hasattr(model.visual_encoder, "model") and hasattr(model.visual_encoder.model, "pos_embed"):
                pos_embed_shape = model.visual_encoder.model.pos_embed.shape
                logger.info(f"✓ TimeSformer có weights: pos_embed shape = {pos_embed_shape}")
                
                if hasattr(model.visual_encoder.model, "time_embed"):
                    time_embed_shape = model.visual_encoder.model.time_embed.shape
                    logger.info(f"✓ TimeSformer có temporal weights: time_embed shape = {time_embed_shape}")
                else:
                    logger.warning("⚠️ TimeSformer không có time_embed weights!")
            else:
                logger.warning("⚠️ TimeSformer không có weights hoặc structure không đúng!")
            
            # Kiểm tra chi tiết về QFormer weights
            if hasattr(model, "query_tokens"):
                logger.info(f"✓ Qformer số query tokens: {model.query_tokens.shape[1]}")
                logger.info(f"✓ Qformer query_tokens stats: mean={model.query_tokens.mean().item():.4f}, std={model.query_tokens.std().item():.4f}")
                
                # Kiểm tra các tham số quan trọng của QFormer
                qformer_has_weights = True
                
                # Kiểm tra self-attention weights
                if hasattr(model.Qformer, "bert") and hasattr(model.Qformer.bert, "encoder"):
                    sample_layer = model.Qformer.bert.encoder.layer[0]
                    if hasattr(sample_layer, "attention"):
                        attn_output = sample_layer.attention.self.query.weight
                        logger.info(f"✓ QFormer self-attention weights: shape={attn_output.shape}, mean={attn_output.mean().item():.4f}")
                    else:
                        qformer_has_weights = False
                        logger.warning("⚠️ QFormer không có attention weights!")
                else:
                    qformer_has_weights = False
                    logger.warning("⚠️ QFormer cấu trúc không đúng hoặc thiếu weights!")
                
                # Kiểm tra cross-attention weights
                if hasattr(model.Qformer, "bert") and hasattr(model.Qformer.bert, "encoder"):
                    for i, layer in enumerate(model.Qformer.bert.encoder.layer):
                        if hasattr(layer, "crossattention"):
                            cross_attn = layer.crossattention.self.query.weight
                            logger.info(f"✓ QFormer cross-attention layer {i}: shape={cross_attn.shape}, mean={cross_attn.mean().item():.4f}")
                            break
                
                if qformer_has_weights:
                    logger.info("✓ QFormer đã có weights cần thiết để evaluate")
                else:
                    logger.warning("⚠️ QFormer có thể CHƯA load weights đúng cách!")
            
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        """
        Xử lý sau khi đánh giá và lưu kết quả caption nếu có
        """
        val_result = kwargs.get("val_result", [])
        split_name = kwargs.get("split_name", "val")
        epoch = kwargs.get("epoch", 0)
        
        logger = logging.getLogger(__name__)
        logger.info(f"=== KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP {split_name.upper()} (EPOCH {epoch}) ===")
        
        # Kiểm tra xem trong kết quả có caption không
        has_captions = False
        if val_result and isinstance(val_result, list):
            for item in val_result:
                if isinstance(item, dict) and "caption" in item:
                    has_captions = True
                    break
        
        # Nếu có caption, lưu vào file
        if has_captions:
            # Lấy result_dir từ registry hoặc sử dụng thư mục hiện tại
            try:
                from lavis.common.registry import registry
                result_dir = registry.get_path("result_dir")
            except:
                result_dir = "results"
                os.makedirs(result_dir, exist_ok=True)
            
            # Tạo tên file với thông tin về split và epoch
            filename = f"caption_results_{split_name}_{epoch}"
            
            # Lưu kết quả
            result_file = self.save_result(val_result, result_dir, filename, remove_duplicate="image_id")
            
            # Hiển thị một số caption để xem nhanh
            logger.info(f"✓ Đã lưu {len(val_result)} caption vào file: {result_file}")
            logger.info("=== MẪU KẾT QUẢ CAPTION ===")
            
            # Hiển thị 5 mẫu caption đầu tiên
            sample_count = min(5, len(val_result))
            for i in range(sample_count):
                if "image_id" in val_result[i] and "caption" in val_result[i]:
                    logger.info(f"Video {val_result[i]['image_id']}: {val_result[i]['caption']}")
        
        # Nếu có agg_metrics, trả về như cũ
        if "agg_metrics" in kwargs:
            return {"agg_metrics": kwargs["agg_metrics"], "best_epoch": kwargs.get("best_epoch", 0)}
        
        return None

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=len(data_loader),
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            ## notify model that sample is empty (error occured)
            if not isinstance(samples, dict):
                samples = {"is_empty":True}

            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, loss_dict = self.train_step(model=model, samples=samples)
                loss /= accum_grad_iters #TODO: not affect loss_dict values for logging

            # after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(**loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file
