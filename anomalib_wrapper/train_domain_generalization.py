import os
from argparse import Namespace, ArgumentParser

import torch
from mmcv import DictAction

from anomalib.config import get_configurable_parameters
from anomalib.models.patchcore import PatchcoreLightning
from .test import test
from .train import train
from .utils.args import add_log_level_arg
from .utils.config import overwrite_conf_from_options


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--target-domain", type=str, help="Target domain to be tested.")
    parser.add_argument("--config", type=str, help="Path to a model config file")
    parser.add_argument("--name", type=str, required=False, default=None, help="Name for logging")
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


def load_miro_feature_extractor(config):
    checkpoint_path = config.model.pretrained_miro_weights
    miro_config = get_configurable_parameters(config_path=os.path.splitext(checkpoint_path)[0] + ".yaml")
    assert config.model.backbone == miro_config.model.backbone
    miro_config.model.supress_feature_extraction = True  # otherwise MIRO will register forward hooks and collect all outputs => huge memory consumption
    model = PatchcoreLightning(miro_config)
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    featurizer: "domainbed.networks.URResNet" = model._miro.featurizer
    return featurizer.network


def train_and_evaluate(target_domain: str, config_path: str, cfg_options=None, name=None):
    config = get_configurable_parameters(config_path=config_path, category_overwrite=cfg_options.get("dataset.category"))
    overwrite_conf_from_options(config, cfg_options)

    backbone = load_miro_feature_extractor(config) if config.model.get("pretrained_miro_weights") else None

    if not name:
        name = "+".join(config.dataset.category) + "/" + target_domain
    cfg_options["dataset.target_domain"] = target_domain  # add to config to be logged to wandb
    # make coreset smaller (divide by number of training domains) for performance reasons
    cfg_options["model.coreset_sampling_ratio"] = config.model.coreset_sampling_ratio / (len(config.dataset.category) - 1)

    if config.model.type == "normal":
        # use anomalies only for test
        for category in config.dataset.custom_mapping.custom_labels:
            for label in config.dataset.custom_mapping.custom_labels[category]:
                if label != "good":
                    cfg_options[f"dataset.custom_mapping.custom_labels.{category}.{label}.split"] = "test"

    train_cfg_options = {
        **cfg_options,
        f"dataset.custom_mapping.custom_labels.{target_domain}": "_delete_",  # delete target domain from mapping
        "dataset.category": [c for c in config.dataset.category if c != target_domain],  # ignore target domain
        "metrics.prefix": "source_",
    }
    trainer = train(None, config_path, train_cfg_options, overwrite_name=name, overwrite_backbone=backbone)

    test_cfg_options = {
        **cfg_options,
        # set all anomaly types of target domain to test
        **{f"dataset.custom_mapping.custom_labels.{target_domain}.{anomaly_type}.split": "test"
           for anomaly_type in config.dataset.custom_mapping.custom_labels.get(target_domain)},
        "dataset.category": target_domain,
        "metrics.prefix": "target_",
    }

    val_cfg_options = {
        **test_cfg_options,
        "logging.logger": [],
    }

    if config.metrics.threshold.get("on_test_image_level", False) and config.metrics.threshold.adaptive:
        # validate again with test images instead of validation patches
        test(config_path, val_cfg_options, model=trainer.model, validate_instead=True, overwrite_name=name)
    test(config_path, test_cfg_options, model=trainer.model, overwrite_name=name)


if __name__ == "__main__":
    args = get_args()
    train_and_evaluate(args.target_domain, args.config, args.cfg_options, args.name)
