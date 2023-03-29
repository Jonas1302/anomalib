"""Towards Total Recall in Industrial Anomaly Detection.

Paper https://arxiv.org/abs/2106.08265.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Tuple, Any, Optional, Dict

from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from jaxtyping import Float
from torch import Tensor

from anomalib.models.components import AnomalyModule, KCenterGreedyBulk, KCenterGreedyOnline, KCenterGreedyOnDemand, KCenterRandom, KCenterAll
from anomalib.models.patchcore.classifier import TransferLearningClassifier, PatchBasedClassifier
from anomalib.models.patchcore.torch_model import PatchcoreModel, LabeledPatchcore
from anomalib.models.patchcore.utils import process_pred_masks, process_label_and_score

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
        num_classes: int,
        pre_trained: bool = True,
        coreset_sampling_ratio: float = 0.1,
        coreset_sampling_mode: str = "bulk",
        num_neighbors: int = 9,
        anomaly_map_with_neighbours: bool = False,
        locally_aware_patch_features: bool = True,
        task: str = "segmentation",
        labeled_coreset: bool = False,
        anomaly_threshold: float = 0.1,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.task = task

        if coreset_sampling_mode == "bulk":
            coreset_sampling_class = KCenterGreedyBulk
        elif coreset_sampling_mode == "online":
            coreset_sampling_class = KCenterGreedyOnline
        elif coreset_sampling_mode == "ondemand":
            coreset_sampling_class = KCenterGreedyOnDemand
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
            num_classes=num_classes,
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
        anomaly_maps: Float[Tensor, "b c w h"]
        anomaly_score: Float[Tensor, "b c"]
        anomaly_maps, anomaly_score = self.model(batch["image"])
        batch["anomaly_maps"] = anomaly_maps
        batch["pred_scores"] = anomaly_score
        batch["label_mapping"] = self.trainer.datamodule.label_mapping  # add for better visualization

        return batch


class MultiStepPatchcore(Patchcore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.model.coreset_sampling, KCenterGreedyOnDemand), f"{self.model.coreset_sampling=}"
        self.training_batches: Dict[str, List[Dict]] = {"good": []}  # dict of batches with labels as keys

    def training_step(self, batch, _batch_idx):
        assert len(batch["label"]) == 1, "train batch size must be 1"
        self.training_batches.setdefault(batch["label_name"][0], []).append(batch)

    def on_validation_start(self):
        for batches in self.training_batches.values():  # note: since we added "good" at creation, it'll be the first entry
            self.model.train()
            for batch in batches:
                self.model(batch["image"], ground_truths=batch.get("mask"), labels=batch.get("label"))
            self.model.eval()
            self.model.calculate_coreset()

class ClassificationPatchcore(Patchcore):
    def __init__(self, num_classes: int, use_threshold: bool = True, *args, **kwargs):
        super().__init__(*args, num_classes=num_classes, use_threshold=use_threshold, **kwargs)
        self.use_threshold = use_threshold
        assert self.use_threshold or num_classes != 1, f"{self.use_threshold=}, {num_classes=}"

    def predict_step(self, batch: Any, batch_idx: int, _dataloader_idx: Optional[int] = None) -> Any:
        anomaly_maps, anomaly_patch_maps = self.model(batch["image"])
        batch["anomaly_maps"] = anomaly_maps

        threshold = self.image_threshold if self.use_threshold else None
        pred_masks, pred_patch_masks = process_pred_masks(anomaly_maps, anomaly_patch_maps, batch, threshold)
        process_label_and_score(anomaly_patch_maps, pred_patch_masks, batch, self.trainer)

        return batch

    def validation_step(self, batch, _):
        return self.predict_step(batch, _)


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

        additional_kwargs = {}
        model_type = hparams.model.get("type", "normal")
        if model_type == "normal":
            cls2 = Patchcore
        elif model_type == "multistep":
            cls2 = MultiStepPatchcore
        elif model_type in ["embedding", "labeled-coreset"]:
            cls2 = ClassificationPatchcore
        elif model_type in ["finetuned-cnn", "transfer-learning"]:
            cls2 = TransferLearningClassifier
        elif model_type == "transfer-learning+mlp":
            cls2 = TransferLearningClassifier
            additional_kwargs["use_mlp"] = True
        elif model_type == "embedding-global-fc":
            cls2 = TransferLearningClassifier
            additional_kwargs["use_global_embedding"] = True
        elif model_type == "embedding-global-mlp":
            cls2 = TransferLearningClassifier
            additional_kwargs["use_mlp"] = True
            additional_kwargs["use_global_embedding"] = True
        elif model_type == "embedding-mlp":
            cls2 = PatchBasedClassifier
        else:
            raise ValueError(f"unknown {model_type=}")

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
            num_classes=hparams.dataset.get("num_classes", None),
            lr=hparams.model.get("lr"),
            hidden_size=hparams.model.get("hidden_size"),
            use_threshold=hparams.model.get("use_threshold"),
            dropout=hparams.model.get("use_dropout"),
            freeze_batch_norm=hparams.model.get("freeze_batch_norm"),
            **additional_kwargs,
        )
        obj.save_hyperparameters(hparams)

        return obj
