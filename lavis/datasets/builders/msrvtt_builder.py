from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.msrvtt_datasets import MSRVTTCaptionDataset

@registry.register_builder("msrvtt")
class MSRVTTBuilder(BaseDatasetBuilder):
    train_dataset_cls = MSRVTTCaptionDataset
    eval_dataset_cls = MSRVTTCaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/defaults_caption.yaml"
    } 