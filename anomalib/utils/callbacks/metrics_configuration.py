"""Metrics Configuration Callback."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging
from typing import List, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY

from anomalib.models.components.base.anomaly_module import AnomalyModule
from anomalib.utils.metrics import create_metric_collection

logger = logging.getLogger(__name__)

__all__ = ["MetricsConfigurationCallback"]


@CALLBACK_REGISTRY
class MetricsConfigurationCallback(Callback):
    """Metrics Configuration Callback."""

    def __init__(
        self,
        task: str = "segmentation",
        image_metrics: Optional[List[str]] = None,
        pixel_metrics: Optional[List[str]] = None,
        prefix: str = "",
    ):
        """Create image and pixel-level AnomalibMetricsCollection.

        This callback creates AnomalibMetricsCollection based on the
            list of strings provided for image and pixel-level metrics.
        After these MetricCollections are created, the callback assigns
        these to the lightning module.

        Args:
            task (str): Task type of the current run.
            image_metrics (Optional[List[str]]): List of image-level metrics.
            pixel_metrics (Optional[List[str]]): List of pixel-level metrics.
        """
        self.task = task
        self.image_metric_names = image_metrics
        self.pixel_metric_names = pixel_metrics
        self.prefix = prefix

    def setup(
        self,
        _trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: Optional[str] = None,  # pylint: disable=unused-argument
    ) -> None:
        """Setup image and pixel-level AnomalibMetricsCollection within Anomalib Model.

        Args:
            _trainer (pl.Trainer): PyTorch Lightning Trainer
            pl_module (pl.LightningModule): Anomalib Model that inherits pl LightningModule.
            stage (Optional[str], optional): fit, validate, test or predict. Defaults to None.
        """
        image_metric_names = [] if self.image_metric_names is None else self.image_metric_names

        pixel_metric_names: List[str]
        if self.pixel_metric_names is None:
            pixel_metric_names = []
        elif self.task == "classification":
            pixel_metric_names = []
            logger.warning(
                "Cannot perform pixel-level evaluation when task type is classification. "
                "Ignoring the following pixel-level metrics: %s",
                self.pixel_metric_names,
            )
        else:
            pixel_metric_names = self.pixel_metric_names

        if isinstance(pl_module, AnomalyModule):
            datamodule = pl_module.trainer.datamodule
            num_classes = datamodule.num_classes if hasattr(datamodule, "num_classes") else None
            if num_classes == 2:
                num_classes = 1  # binary classification uses only one output
            pl_module.image_metrics = create_metric_collection(image_metric_names, f"{self.prefix}image_", num_classes)
            pl_module.pixel_metrics = create_metric_collection(pixel_metric_names, f"{self.prefix}pixel_", None)

            if pl_module.image_threshold is not None:
                pl_module.image_metrics.set_threshold(pl_module.image_threshold.value)
                pl_module.pixel_metrics.set_threshold(pl_module.pixel_threshold.value)
