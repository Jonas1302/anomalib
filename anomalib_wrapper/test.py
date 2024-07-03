"""Test This script performs inference on the test dataset and saves the output visualizations into a directory."""
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

from pytorch_lightning import Trainer, seed_everything

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks
from anomalib.utils.loggers import get_experiment_logger, AnomalibWandbLogger
from .utils.args import add_log_level_arg
from .utils.config import overwrite_conf_from_options


def get_args() -> Namespace:
    """Get CLI arguments.

    Returns:
        Namespace: CLI arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="stfpm", help="Name of the algorithm to train/test")
    parser.add_argument("--config", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--weight_path", type=str, default="weights/model.ckpt")
    parser.add_argument("--openvino", type=bool, default=False)
    add_log_level_arg(parser)

    return parser.parse_args()


def test(config, cfg_options, overwrite_name=None, model=None, validate_instead=False):
    """Test an anomaly classification and segmentation model that is initially trained via `tools/train.py`.

    The script is able to write the results into both filesystem and a logger such as Tensorboard.
    """
    config = get_configurable_parameters(
        model_name=model,
        config_path=config,
    )
    overwrite_conf_from_options(config, cfg_options)

    if config.project.seed is not None:
        seed_everything(config.project.seed)
    if config.visualization.image_save_path:  # put all images in a subfolder
        config.visualization.image_save_path = os.path.join(config.visualization.image_save_path,
                                                            overwrite_name if overwrite_name else config.dataset.category)

    datamodule = get_datamodule(config)
    if model is None:
        model = get_model(config)

    callbacks = get_callbacks(config)

    experiment_logger = get_experiment_logger(config, overwrite_name)

    trainer = Trainer(callbacks=callbacks, logger=experiment_logger, **config.trainer)
    if validate_instead:
        trainer.validate(model=model, datamodule=datamodule)
    else:
        trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    args = get_args()
    test(args.model, args.config, args.weight_path)
