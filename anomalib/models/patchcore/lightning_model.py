"""Towards Total Recall in Industrial Anomaly Detection.

Paper https://arxiv.org/abs/2106.08265.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Tuple, Union, Dict, Any, Optional

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from torch import Tensor

from anomalib.models.components import AnomalyModule, KCenterGreedyBulk, KCenterGreedyOnline, KCenterRandom, KCenterAll
from anomalib.models.patchcore.classifier import ResnetClassifier
from anomalib.models.patchcore.torch_model import PatchcoreModel, LabeledPatchcore

logger = logging.getLogger(__name__)


@MODEL_REGISTRY
class Patchcore(AnomalyModule):
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
        pre_trained: bool = True,
        coreset_sampling_ratio: float = 0.1,
        coreset_sampling_mode: str = "bulk",
        num_neighbors: int = 9,
        anomaly_map_with_neighbours: bool = False,
        locally_aware_patch_features: bool = True,
        task: str = "segmentation",
        labeled_coreset: bool = False,
        anomaly_threshold: float = 0.1,
        most_common_anomaly_instead_of_highest_score: bool = True,
    ) -> None:
        super().__init__()
        self.task = task

        if coreset_sampling_mode == "bulk":
            coreset_sampling_class = KCenterGreedyBulk
        elif coreset_sampling_mode == "online":
            coreset_sampling_class = KCenterGreedyOnline
        elif coreset_sampling_mode == "random":
            coreset_sampling_class = KCenterRandom
        elif coreset_sampling_mode == "all":
            coreset_sampling_class = KCenterAll
        else:
            raise ValueError(f"unknown coreset subsampling mode: {coreset_sampling_mode}")

        patchcore_class = LabeledPatchcore if labeled_coreset else PatchcoreModel

        self.model: PatchcoreModel = patchcore_class(
            input_size=input_size,
            backbone=backbone,
            pre_trained=pre_trained,
            layers=layers,
            num_neighbors=num_neighbors,
            anomaly_map_with_neighbours=anomaly_map_with_neighbours,
            locally_aware_patch_features=locally_aware_patch_features,
            coreset_sampling=coreset_sampling_class(coreset_sampling_ratio),
            anomaly_threshold=anomaly_threshold,
            most_common_anomaly_instead_of_highest_score=most_common_anomaly_instead_of_highest_score,
        )
        self.most_common_anomaly_instead_of_highest_score = most_common_anomaly_instead_of_highest_score

    def configure_optimizers(self) -> None:
        """Configure optimizers.

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
        self.model(batch["image"], ground_truths=batch.get("mask"), labels=batch.get("label"))

    def on_validation_start(self) -> None:
        """Apply subsampling to the embedding collected from the training set."""
        # NOTE: Previous anomalib versions fit subsampling at the end of the epoch.
        #   This is not possible anymore with PyTorch Lightning v1.4.0 since validation
        #   is run within train epoch.
        logger.info("Applying core-set subsampling to get the embedding.")
        self.model.calculate_coreset()

    def validation_step(self, batch, _):  # pylint: disable=arguments-differ
        """Get batch of anomaly maps from input image batch.

        Args:
            batch (Dict[str, Any]): Batch containing image filename,
                                    image, label and mask
            _ (int): Batch Index

        Returns:
            Dict[str, Any]: Image filenames, test images, GT and predicted label/masks
        """
        anomaly_maps, anomaly_score = self.model(batch["image"])
        batch["anomaly_maps"] = anomaly_maps
        batch["pred_scores"] = anomaly_score

        return batch


class ClassificationPatchcore(Patchcore):

    def predict_step(self, batch: Any, batch_idx: int, _dataloader_idx: Optional[int] = None) -> Any:
        """Step function called during :meth:`~pytorch_lightning.trainer.trainer.Trainer.predict`.

        By default, it calls :meth:`~pytorch_lightning.core.lightning.LightningModule.forward`.
        Override to add any processing logic.

        Args:
            batch (Tensor): Current batch
            batch_idx (int): Index of current batch
            _dataloader_idx (int): Index of the current dataloader

        Return:
            Predicted output
        """
        anomaly_maps, anomaly_score, anomaly_patch_maps = self.model(batch["image"], get_anomaly_patch_map=True)
        outputs = batch
        outputs["anomaly_maps"] = anomaly_maps
        outputs["pred_scores"] = anomaly_score

        label_mapping = self.trainer.datamodule.label_mapping  # add for better visualization
        outputs["label_mapping"] = label_mapping

        if len(self.image_threshold.value.shape) > 0:
            pred_masks = outputs["anomaly_maps"].permute(0, 2, 3, 1) >= self.image_threshold.value
            pred_masks = pred_masks.permute(0, 3, 1, 2)  # permute back to original shape
            pred_patch_masks = anomaly_patch_maps.permute(0, 2, 3, 1) >= self.image_threshold.value
            pred_patch_masks = pred_patch_masks.permute(0, 3, 1, 2)
        else:
            pred_masks = outputs["anomaly_maps"] >= self.image_threshold.value
            pred_patch_masks = anomaly_patch_maps >= self.image_threshold.value
        outputs["pred_masks"] = pred_masks

        pred_scores_normed = outputs["pred_scores"] - self.image_threshold.value
        outputs["pred_labels"] = []

        for i in range(len(pred_masks)):
            if self.most_common_anomaly_instead_of_highest_score:
                bincount = torch.stack([pred_patch_masks[i, j].sum() for j in range(len(pred_patch_masks[i]))])
                if bincount[1:].any():  # any anomalous patches
                    label = bincount[1:].argmax().item() + 1
                else:
                    label = 0
            else:
                if pred_scores_normed[i].max() < 0:  # all anomaly scores are below threshold => good
                    label = 0
                else:
                    label = pred_scores_normed[i].argmax().item()
            outputs["pred_labels"].append(label_mapping[label])
            outputs["pred_scores"][i, torch.arange(0, len(pred_masks[i])) != label] = 0  # set all to zero except label

        return outputs


class PatchcoreLightning:
    """PatchcoreLightning Module to train PatchCore algorithm.

    Args:
        hparams (Union[DictConfig, ListConfig]): Model params
    """
    def __new__(cls, hparams):
        # this class is basically just a factory method disguised as class
        # since it is woven into other complex parts of anomalib's code, I think it's easier
        # to mangle with the object's construction
        assert cls == PatchcoreLightning, "PatchcoreLightning cannot be subclassed (subclass would be ignored)"

        task = hparams.dataset.get("task", "segmentation")
        if task == "segmentation":
            cls2 = Patchcore
        elif task == "classification":
            model_type = hparams.model.get("type", "embedding")
            if model_type == "embedding":
                cls2 = ClassificationPatchcore
            elif model_type == "cnn":
                cls2 = ResnetClassifier
            else:
                raise ValueError(f"unknown model type {model_type}")
        else:
            raise ValueError(f"unknown task {task}")

        obj = cls2(
            input_size=hparams.model.input_size,
            backbone=hparams.model.backbone,
            layers=hparams.model.layers,
            pre_trained=hparams.model.pre_trained,
            coreset_sampling_ratio=hparams.model.coreset_sampling_ratio,
            coreset_sampling_mode=hparams.model.get("coreset_sampling_mode", "bulk"),
            num_neighbors=hparams.model.num_neighbors,
            anomaly_map_with_neighbours=hparams.model.get("anomaly_map_with_neighbours", False),
            locally_aware_patch_features=hparams.model.get("locally_aware_patch_features", True),
            task=hparams.dataset.get("task", "segmentation"),
            labeled_coreset=hparams.model.get("labeled_coreset", False),
            anomaly_threshold=hparams.model.get("anomaly_threshold", 0.1),
            most_common_anomaly_instead_of_highest_score=hparams.model.get("most_common_anomaly_instead_of_highest_score", True),
            num_classes=hparams.dataset.get("num_classes", None),
            lr=hparams.model.get("lr"),
        )
        obj.save_hyperparameters(hparams)

        return obj
