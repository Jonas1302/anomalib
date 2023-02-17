"""Towards Total Recall in Industrial Anomaly Detection.

Paper https://arxiv.org/abs/2106.08265.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Tuple, Union, Dict

from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.cli import MODEL_REGISTRY

from anomalib.models.components import AnomalyModule, KCenterGreedyBulk, KCenterGreedyOnline, KCenterRandom, KCenterAll
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

        if labeled_coreset:
            patchcore_class = LabeledPatchcore
            self.image_threshold = None
            self.pixel_threshold = None
        else:
            patchcore_class = PatchcoreModel

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
        label_mapping = self.trainer.datamodule.label_mapping

        anomaly_maps, anomaly_score = self.model(batch["image"])
        batch["anomaly_maps"] = anomaly_maps
        batch["pred_scores"] = anomaly_score
        batch["pred_labels"] = [label_mapping[index.argmax().item()] for index in anomaly_score]
        batch["label_mapping"] = label_mapping  # add for better visualization

        return batch

    def _compute_adaptive_threshold(self, outputs):
        # only calculate threshold if not classifying
        if self.task != "classification":
            super()._compute_adaptive_threshold(outputs)


class PatchcoreLightning(Patchcore):
    """PatchcoreLightning Module to train PatchCore algorithm.

    Args:
        hparams (Union[DictConfig, ListConfig]): Model params
    """

    def __init__(self, hparams) -> None:
        super().__init__(
            input_size=hparams.model.input_size,
            backbone=hparams.model.backbone,
            layers=hparams.model.layers,
            pre_trained=hparams.model.pre_trained,
            coreset_sampling_ratio=hparams.model.coreset_sampling_ratio,
            coreset_sampling_mode=hparams.model.get("coreset_sampling_mode", "bulk"),
            num_neighbors=hparams.model.num_neighbors,
            anomaly_map_with_neighbours=hparams.model.get("anomaly_map_with_neighbours", False),
            locally_aware_patch_features=hparams.model.get("locally_aware_patch_features", True),
            task=hparams.dataset.get("task", "default"),
            labeled_coreset=hparams.model.get("labeled_coreset", False),
            anomaly_threshold=hparams.model.get("anomaly_threshold", 0.1),
            most_common_anomaly_instead_of_highest_score=hparams.model.get("most_common_anomaly_instead_of_highest_score", True),
        )
        self.hparams: Union[DictConfig, ListConfig]  # type: ignore
        self.save_hyperparameters(hparams)
