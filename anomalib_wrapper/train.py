"""Anomalib Traning Script.

This script reads the name of the model or config file from command
line, train/test the anomaly model to get quantitative and qualitative
results.
"""
import json
import os
# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union, Optional, Dict, Any, List

from mmcv import DictAction

from pytorch_lightning import seed_everything

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.deploy.trainer import Trainer
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks, LoadModelCallback
from anomalib.utils.loggers import get_experiment_logger, AnomalibWandbLogger

from .utils.args import add_log_level_arg
from .utils.config import overwrite_conf_from_options


def get_args() -> Namespace:
    """Get command line arguments.

    :returns: Namespace: List of arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="padim", help="Name of the algorithm to train/test")
    parser.add_argument("--config", type=str, required=False, help="Path to a model config file")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        default={},
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file. For example, "
             "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'",
    )
    add_log_level_arg(parser)

    return parser.parse_args()


def train(model: Optional[str], config_path: Union[str, Path, None] = None, cfg_options: Optional[Dict[str, Any]] = None,
          overwrite_name: Optional[str] = None, overwrite_backbone=None):
    """Train an anomaly classification or segmentation model based on a provided configuration file."""
    if cfg_options is None:
        cfg_options = {}
    config = get_configurable_parameters(model_name=model, config_path=config_path,
                                         category_overwrite=cfg_options.get("dataset.category"))
    overwrite_conf_from_options(config, cfg_options)

    if config.model.type == "normal":
        config.model.normalization_method = "min_max"
    else:
        config.model.normalization_method = None

    if config.project.seed is not None:
        seed_everything(config.project.seed)
    if config.visualization.image_save_path:  # put all images in a subfolder
        if overwrite_name:
            folder_name = overwrite_name
        elif isinstance(config.dataset.category, str):
            folder_name = config.dataset.category
        else:  # category is list
            folder_name = "+".join(config.dataset.category)
        config.visualization.image_save_path = os.path.join(config.visualization.image_save_path, folder_name)

    experiment_logger = get_experiment_logger(config, overwrite_name)

    datamodule = get_datamodule(config)
    config.dataset.num_classes = datamodule.num_classes
    config.dataset.num_categories = datamodule.num_categories
    model = get_model(config)
    if overwrite_backbone:
        model.overwrite_backbone(overwrite_backbone)
    if config.model.get("type") == "embedding-mlp":
        datamodule.set_train_embedding_extractor(model.extract_embeddings, config.dataset.image_size[0] // 8,
                                                 config.model.anomaly_threshold,
                                                 config.dataset.custom_mapping.use_coreset_subsampling)

    trainer = Trainer(**config.trainer, logger=experiment_logger, callbacks=get_callbacks(config))
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)

    return trainer


if __name__ == "__main__":
    args = get_args()
    train(args.model, args.config, args.cfg_options)
