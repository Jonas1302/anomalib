"""Anomaly Visualization."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import cv2
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.segmentation import mark_boundaries
from torch import Tensor

from anomalib.data.utils import read_image
from anomalib.post_processing.post_process import (
    add_anomalous_label,
    add_normal_label,
    superimpose_anomaly_map, add_class_label,
)


@dataclass
class ImageResult:
    """Collection of data needed to visualize the predictions for an image."""

    image: np.ndarray
    pred_score: float
    pred_label: str = None
    anomaly_maps: Optional[np.ndarray] = None
    gt_mask: Optional[np.ndarray] = None
    pred_mask: Optional[np.ndarray] = None
    pred_mask_image_threshold: Optional[np.ndarray] = None
    label: Optional[int] = None
    label_mapping: Optional[Dict[int, str]] = None

    heat_map: Optional[np.ndarray] = field(init=False, default=None)
    segmentations: Optional[np.ndarray] = field(init=False, default=None)
    heat_map_with_segmentations: Optional[np.ndarray] = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Generate heatmap overlay and segmentations, convert masks to images."""
        # create heat maps
        if self.anomaly_maps is not None:
            if len(self.anomaly_maps.shape) == 2:
                self.anomaly_maps = np.expand_dims(self.anomaly_maps, 0)
            self.heat_map = np.stack([
                superimpose_anomaly_map(anomaly_map, self.image)
                for anomaly_map in self.anomaly_maps
            ])
        # create segmentations
        if self.pred_mask is not None and self.pred_mask.max() <= 1.0:
            self.pred_mask *= 255
            if len(self.pred_mask.shape) == 2:
                self.segmentations = np.expand_dims(self.segmentations, 0)
            self.segmentations = np.stack([self._create_segmentations(self.image, pred_mask) for pred_mask in self.pred_mask])
            if self.anomaly_maps is not None:
                for i in range(len(self.segmentations)):
                    self.heat_map_with_segmentations = np.stack([
                        self._create_segmentations(self.heat_map[i], self.pred_mask[i], color=(0, 1, 0))
                        for i in range(len(self.heat_map))
                    ])
        # normalize masks from [0, 1] to [0, 255]
        if self.pred_mask_image_threshold is not None and self.pred_mask_image_threshold.max() <= 1.0:
            self.pred_mask_image_threshold *= 255
        if self.gt_mask is not None and self.gt_mask.max() <= 1.0:
            self.gt_mask *= 255

    @staticmethod
    def _create_segmentations(image, pred_mask, color=(1, 0, 0)):
        segmentations = mark_boundaries(image, pred_mask, color=color, mode="thick")
        if segmentations.max() <= 1.0:
            segmentations = (segmentations * 255).astype(np.uint8)
        return segmentations


class Visualizer:
    """Class that handles the logic of composing the visualizations.

    Args:
        mode (str): visualization mode, either "full" or "simple"
        task (str): task type, either "segmentation" or "classification"
    """

    def __init__(self, mode: str, task: str) -> None:
        if mode not in ["full", "simple"]:
            raise ValueError(f"Unknown visualization mode: {mode}. Please choose one of ['full', 'simple']")
        self.mode = mode
        if task not in ["classification", "segmentation"]:
            raise ValueError(f"Unknown task type: {mode}. Please choose one of ['classification', 'segmentation']")
        self.task = task

    def visualize_batch(self, batch: Dict) -> Iterator[np.ndarray]:
        """Generator that yields a visualization result for each item in the batch.

        Args:
            batch (Dict): Dictionary containing the ground truth and predictions of a batch of images.

        Returns:
            Generator that yields a display-ready visualization for each image.
        """
        batch_size, _num_channels, height, width = batch["image"].size()
        for i in range(batch_size):
            if "image_visualization" in batch:
                image = batch["image_visualization"][i]
                if isinstance(image, Tensor):
                    image = image.cpu().numpy()
            else:
                # re-read because `batch["image"]` was normalized
                image = read_image(path=batch["image_path"][i], image_size=(height, width))

            pred_score = batch["pred_scores"][i]
            if len(pred_score.shape) == 1:
                pred_score = pred_score.argmax()
            pred_label = batch["pred_labels"][i]

            image_result = ImageResult(
                image=image,
                pred_score=pred_score.cpu().numpy().item(),
                pred_label=pred_label.cpu().numpy().item() if isinstance(pred_label, torch.Tensor) else pred_label,
                anomaly_maps=batch["anomaly_maps"][i].cpu().numpy() if "anomaly_maps" in batch else None,
                pred_mask=batch["pred_masks"][i].squeeze().int().cpu().numpy() if "pred_masks" in batch else None,
                pred_mask_image_threshold=batch["pred_masks_image_threshold"][i].squeeze().int().cpu().numpy() if "pred_masks_image_threshold" in batch else None,
                gt_mask=batch["mask"][i].squeeze().int().cpu().numpy() if "mask" in batch else None,
                label=batch["label"][i].item() if "label" in batch else None,
                label_mapping=batch["label_mapping"],
            )
            yield self.visualize_image(image_result)

    def visualize_image(self, image_result: ImageResult) -> np.ndarray:
        """Generate the visualization for an image.

        Args:
            image_result (ImageResult): GT and Prediction data for a single image.

        Returns:
            The full or simple visualization for the image, depending on the specified mode.
        """
        if self.mode == "full":
            return self._visualize_full(image_result)
        if self.mode == "simple":
            return self._visualize_simple(image_result)
        raise ValueError(f"Unknown visualization mode: {self.mode}")

    def _add_label(self, image: np.ndarray, image_result: ImageResult) -> np.ndarray:
        if self.task == "classification":
            return add_class_label(image, image_result.pred_label, image_result.pred_score == image_result.label)
        elif image_result.pred_label:
            return add_anomalous_label(image, image_result.pred_score)
        else:
            return add_normal_label(image, 1 - image_result.pred_score)

    def _visualize_full(self, image_result: ImageResult) -> np.ndarray:
        """Generate the full set of visualization for an image.

        The full visualization mode shows a grid with subplots that contain the original image, the GT mask (if
        available), the predicted heat map, the predicted segmentation mask (if available), and the predicted
        segmentations (if available).

        Args:
            image_result (ImageResult): GT and Prediction data for a single image.

        Returns:
            An image showing the full set of visualizations for the input image.
        """
        visualization = ImageGrid()
        visualization.add_image(self._add_label(image_result.image, image_result), "Image")  # TODO add label only if heat_map is None
        if self.task == "segmentation" and image_result.gt_mask is not None:
            visualization.add_image(image=image_result.gt_mask, color_map="gray", title="Ground Truth")
        if image_result.heat_map is not None and len(image_result.heat_map) > 2:
            # use combined heat map with segmentations to reduce size of visualization image
            for i in range(len(image_result.heat_map_with_segmentations)):
                visualization.add_image(self._add_label(image_result.heat_map_with_segmentations[i], image_result),
                                        f"Predicted Heat Map ({image_result.label_mapping[i]})")
        else:
            if image_result.heat_map is not None:
                for i in range(len(image_result.heat_map)):
                    visualization.add_image(self._add_label(image_result.heat_map[i], image_result),
                                            f"Predicted Heat Map ({image_result.label_mapping[i]})")
            if self.task == "classification" and image_result.pred_mask_image_threshold is not None:
                visualization.add_image(image=image_result.pred_mask_image_threshold,
                                        color_map="gray", title="Predicted Mask (Image Threshold)")
            if image_result.pred_mask is not None:
                for i in range(len(image_result.pred_mask)):
                    visualization.add_image(image=image_result.pred_mask[i], color_map="gray", title="Predicted Mask")
                    visualization.add_image(image=image_result.segmentations[i], title=f"Segmentation Result")

        return visualization.generate()

    def _visualize_simple(self, image_result: ImageResult) -> np.ndarray:
        """Generate a simple visualization for an image.

        The simple visualization mode only shows the model's predictions in a single image.

        Args:
            image_result (ImageResult): GT and Prediction data for a single image.

        Returns:
            An image showing the simple visualization for the input image.
        """
        if self.task == "segmentation":
            heat_map = image_result.heat_map[image_result.pred_score] if len(image_result.heat_map) > 1 else image_result.heat_map[0]
            visualization = mark_boundaries(heat_map, image_result.pred_mask, color=(1, 0, 0), mode="thick")
            visualization = (visualization * 255).astype(np.uint8)
            return self._add_label(visualization, image_result)
        if self.task == "classification":
            return self._add_label(image_result.image, image_result)
        raise ValueError(f"Unknown task type: {self.task}")

    @staticmethod
    def show(title: str, image: np.ndarray, delay: int = 0) -> None:
        """Show an image on the screen.

        Args:
            title (str): Title that will be given to the window showing the image.
            image (np.ndarray): Image that will be shown in the window.
            delay (int): Delay in milliseconds to wait for keystroke. 0 for infinite.
        """
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow(title, image)
        cv2.waitKey(delay)
        cv2.destroyAllWindows()

    @staticmethod
    def save(file_path: Path, image: np.ndarray) -> None:
        """Save an image to the file system.

        Args:
            file_path (Path): Path to which the image will be saved.
            image (np.ndarray): Image that will be saved to the file system.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(file_path), image)


class ImageGrid:
    """Helper class that compiles multiple images into a grid using subplots.

    Individual images can be added with the `add_image` method. When all images have been added, the `generate` method
    must be called to compile the image grid and obtain the final visualization.
    """

    def __init__(self):
        self.images: List[Dict] = []
        self.figure: matplotlib.figure.Figure
        self.axis: np.ndarray

    def add_image(self, image: np.ndarray, title: Optional[str] = None, color_map: Optional[str] = None) -> None:
        """Add an image to the grid.

        Args:
          image (np.ndarray): Image which should be added to the figure.
          title (str): Image title shown on the plot.
          color_map (Optional[str]): Name of matplotlib color map used to map scalar data to colours. Defaults to None.
        """
        image_data = dict(image=image, title=title, color_map=color_map)
        self.images.append(image_data)

    def generate(self) -> np.ndarray:
        """Generate the image.

        Returns:
            Image consisting of a grid of added images and their title.
        """
        num_cols = len(self.images)
        figure_size = (num_cols * 5, 5)
        self.figure, self.axis = plt.subplots(1, num_cols, figsize=figure_size)
        self.figure.subplots_adjust(right=0.9)

        axes = self.axis if isinstance(self.axis, np.ndarray) else np.array([self.axis])
        for axis, image_dict in zip(axes, self.images):
            axis.axes.xaxis.set_visible(False)
            axis.axes.yaxis.set_visible(False)
            axis.imshow(image_dict["image"], image_dict["color_map"], vmin=0, vmax=255)
            if image_dict["title"] is not None:
                axis.title.set_text(image_dict["title"])
        self.figure.canvas.draw()
        # convert canvas to numpy array to prepare for visualization with opencv
        img = np.frombuffer(self.figure.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.figure.canvas.get_width_height()[::-1] + (3,))
        plt.close(self.figure)
        return img
