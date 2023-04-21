from collections import OrderedDict
from typing import Tuple, List, Dict, Callable

import sconf
import torch
from jaxtyping import Float
from torch import Tensor

from anomalib.models import AnomalyModule
from domainbed.algorithms import MIRO as _MIRO


def remove_all_forward_hooks(model: torch.nn.Module) -> None:
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks: Dict[int, Callable] = OrderedDict()
            remove_all_forward_hooks(child)


class MIRO(AnomalyModule):
    """PatchcoreLightning Module to train PatchCore algorithm.

    Args:
        input_size (Tuple[int, int]): Size of the model input.
        backbone (str): Backbone CNN network
        layers (List[str]): Layers to extract features from the backbone CNN
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
        coreset_sampling_ratio (float, optional): Coreset sampling ratio to subsample embedding.
            Defaults to 0.1.
        num_neighbors (int, optional): Number of nearest neighbors. Defaults to 9.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        backbone: str,
        layers: List[str],
        num_classes: int,
        num_categories: int,
        lr: float,
        dropout: float,
        pre_trained: bool = True,
        coreset_sampling_ratio: float = 0.1,
        coreset_sampling_mode: str = "bulk",
        num_neighbors: int = 9,
        anomaly_map_with_neighbours: bool = False,
        locally_aware_patch_features: bool = True,
        task: str = "segmentation",
        labeled_coreset: bool = False,
        anomaly_threshold: float = 0.1,
        supress_feature_extraction: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes

        self._miro = _MIRO(
            input_shape=(3, *input_size),
            num_classes=num_classes,
            num_domains=num_categories,
            hparams=sconf.Config({
                "feat_layers": "stem_block",  # ["stem_block", "block", None]
                "optimizer": "adam",  # ["adam", "sgd", "adamw"]
                "model": backbone,  # "resnet50"
                "pretrained": True,  # bool
                "resnet_dropout": dropout if dropout else 0.,  # 0.
                "ld": 0.1,  # Union[ScalarFloat, float] (= lambda?)
                "lr": lr,  # 5e-5
                "lr_mult": 10.,
                "weight_decay": 0.,  # float
            })
        )

        if supress_feature_extraction:
            remove_all_forward_hooks(self._miro)

    def configure_optimizers(self) -> None:
        """Configure optimizers. Is done internally when calling `self.training_step(...)`.

        Returns:
            None: Do not set optimizers by returning None.
        """
        return None

    def training_step(self, batch, _batch_idx):  # pylint: disable=arguments-differ
        """Generate feature embedding of the batch.

        Args:
            batch (Dict[str, Any]): Batch containing image filename, image, label and mask
            _batch_idx (int): Batch Index

        Returns:
            Dict[str, np.ndarray]: Embedding Vector
        """
        # note: the default implementation uses always the exact same amount of images from each domain
        result = self._miro.update(batch["image"], batch["label"], self.trainer.datamodule.train_data.images_per_class)
        self.log(f"loss", result["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch["label"]))
        self.log(f"reg_loss", result["reg_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch["label"]))

    def validation_step(self, batch, _, log=True):  # pylint: disable=arguments-differ
        with torch.no_grad():
            logits: Float[Tensor, "b c"]
            logits, loss, reg_loss = self._miro.run(batch["image"], batch["label"], self.trainer.datamodule.val_data.images_per_class)

        if self.num_classes == 1:
            scores = torch.sigmoid(logits.squeeze(dim=1))
        else:
            scores = torch.softmax(logits, dim=1)

        if log:
            self.log(f"val_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch["label"]))
            self.log(f"val_reg_loss", reg_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch["label"]))

        batch["loss"] = loss
        label_mapping = self.trainer.datamodule.label_mapping
        batch["label_mapping"] = label_mapping
        batch["pred_scores"] = scores
        batch["pred_labels"] = scores >= self.image_threshold.value

        return batch

    def predict_step(self, batch, batch_idx: int, _dataloader_idx=None):
        return self.validation_step(batch, batch_idx, log=False)
