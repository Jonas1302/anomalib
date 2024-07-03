from typing import Optional, Dict, Any

from anomalib.config import update_input_size_config, update_nncf_config
from omegaconf import DictConfig, OmegaConf


def overwrite_conf_from_options(
    config: DictConfig, cfg_options: Optional[Dict[str, Any]]
):
    if cfg_options is not None:
        for option, value in cfg_options.items():
            if value == "_delete_":
                sub_config = config
                for name in option.split(".")[:-1]:
                    sub_config = sub_config[name]
                del sub_config[option.split(".")[-1]]
            else:
                OmegaConf.update(config, option, value)
        update_input_size_config(config)
        update_nncf_config(config)
