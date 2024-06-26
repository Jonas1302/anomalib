"""MVTec AD Dataset (CC BY-NC-SA 4.0).

Description:
    This script contains PyTorch Dataset, Dataloader and PyTorch
        Lightning DataModule for the MVTec AD dataset.

    If the dataset is not on the file system, the script downloads and
        extracts the dataset and create PyTorch data objects.

License:
    MVTec AD dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0)(https://creativecommons.org/licenses/by-nc-sa/4.0/).

Reference:
    - Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger:
      The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for
      Unsupervised Anomaly Detection; in: International Journal of Computer Vision
      129(4):1038-1059, 2021, DOI: 10.1007/s11263-020-01400-4.

    - Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger: MVTec AD —
      A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection;
      in: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
      9584-9592, 2019, DOI: 10.1109/CVPR.2019.00982.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import tarfile
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List, Literal
from urllib.request import urlretrieve

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf, DictConfig, ListConfig
from pandas.core.frame import DataFrame
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import VisionDataset

from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import DownloadProgressBar, hash_check, read_image
from anomalib.data.utils.split import (
    create_validation_set_from_test_set,
    split_normal_images_in_train_set,
)
from anomalib.utils.typing import EmbeddingExtractor
from anomalib.models.components import KCenterGreedyBulk
from anomalib.pre_processing import PreProcessor

logger = logging.getLogger(__name__)


def make_mvtec_dataset(
    path: Path,
    categories: Union[str, List[str]],
    split: Optional[str] = None,
    split_ratio: float = 0.1,
    seed: Optional[int] = None,
    create_validation_set: bool = False,
    custom_mapping: Optional[DictConfig] = None,
    binary_label_indices: bool = True,
) -> Tuple[DataFrame, Dict[int, str]]:
    """Create MVTec AD samples by parsing the MVTec AD data file structure.

    The files are expected to follow the structure:
        path/to/dataset/split/category/image_filename.png
        path/to/dataset/ground_truth/category/mask_filename.png

    This function creates a dataframe to store the parsed information based on the following format:
    |---|---------------|-------|---------|---------------|---------------------------------------|-------------|
    |   | path          | split | label   | image_path    | mask_path                             | label_index |
    |---|---------------|-------|---------|---------------|---------------------------------------|-------------|
    | 0 | datasets/name |  test |  defect |  filename.png | ground_truth/defect/filename_mask.png | 1           |
    |---|---------------|-------|---------|---------------|---------------------------------------|-------------|

    Args:
        path (Path): Path to dataset
        categories (Union[str, List[str]]): name of one or multiple categories
        split (str, optional): Dataset split (ie., either train or test). Defaults to None.
        split_ratio (float, optional): Ratio to split normal training images and add to the
            test set in case test set doesn't contain any normal images.
            Defaults to 0.1.
        seed (int, optional): Random seed to ensure reproducibility when splitting. Defaults to 0.
        create_validation_set (bool, optional): Boolean to create a validation set from the test set.
            MVTec AD dataset does not contain a validation set. Those wanting to create a validation set
            could set this flag to ``True``.
        custom_mapping (DictConfig, optional): Config which can be used to overwrite the default labels of an
            anomaly type or to ignore it.
        binary_label_indices (bool): whether to use only 0 (normal) and 1 (anomalous) for label indices or
            a separate number for each anomaly type.

    Examples:
        The following example shows how to get training samples from MVTec AD bottle category:

        >>> root = Path('./MVTec')
        >>> category = 'bottle'
        >>> path = root / category
        >>> path
        PosixPath('MVTec/bottle')

        >>> samples = make_mvtec_dataset(path, split='train', split_ratio=0.1, seed=0)
        >>> samples.head()
           path         split label image_path                           mask_path                   label_index
        0  MVTec/bottle train good MVTec/bottle/train/good/105.png MVTec/bottle/ground_truth/good/105_mask.png 0
        1  MVTec/bottle train good MVTec/bottle/train/good/017.png MVTec/bottle/ground_truth/good/017_mask.png 0
        2  MVTec/bottle train good MVTec/bottle/train/good/137.png MVTec/bottle/ground_truth/good/137_mask.png 0
        3  MVTec/bottle train good MVTec/bottle/train/good/152.png MVTec/bottle/ground_truth/good/152_mask.png 0
        4  MVTec/bottle train good MVTec/bottle/train/good/109.png MVTec/bottle/ground_truth/good/109_mask.png 0

    Returns:
        DataFrame: an output dataframe containing samples for the requested split (ie., train or test)
    """
    if isinstance(categories, str):
        categories = [categories]

    samples_list = []
    for category in categories:
        category_path = path / category
        samples_list.extend(sorted([(str(category_path), category, *filename.parts[-3:])
                                    for filename in category_path.glob("**/*.png")]))

    if len(samples_list) == 0:
        raise RuntimeError(f"Found 0 images in {path} for {categories=}")

    samples = pd.DataFrame(samples_list, columns=["path", "category", "split", "label", "image_path"])
    samples = samples[samples.split != "ground_truth"]

    # Create mask_path column
    samples["mask_path"] = (
        samples.path
        + "/ground_truth/"
        + samples.label
        + "/"
        + samples.image_path.str.rstrip("png").str.rstrip(".")
        + "_mask.png"
    )

    # Modify image_path column by converting to absolute path
    samples["image_path"] = samples.path + "/" + samples.split + "/" + samples.label + "/" + samples.image_path

    # Split the normal images in training set if test set doesn't
    # contain any normal images. This is needed because AUC score
    # cannot be computed based on 1-class
    if sum((samples.split == "test") & (samples.label == "good")) == 0:
        samples = split_normal_images_in_train_set(samples, split_ratio, seed)

    # Good images don't have mask
    samples.loc[samples.label == "good", "mask_path"] = ""
    # store original label in case it's overwritten
    samples["original_label"] = samples.label

    if any([isinstance(label, DictConfig) for category in categories for label in custom_mapping.custom_labels[category].values()]):
        # new format
        samples, label_mapping = _apply_custom_mapping_v2(samples, categories, custom_mapping, binary_label_indices)
    elif binary_label_indices:
        # Create label index for normal (0) and anomalous (1) images.
        samples.loc[(samples.label == "good"), "label_index"] = 0
        samples.loc[(samples.label != "good"), "label_index"] = 1
        label_mapping = {0: "normal", 1: "anomalous"}

        if custom_mapping:  # apply custom changes to the dataset
            for class_label, class_mode in custom_mapping.custom_labels[category].items():
                assert samples.label.isin([class_label]).any(), f"category '{category}' does not contain given label '{class_label}'"
                if class_mode == "ignore":
                    samples = samples[samples.label != class_label]
                elif class_mode == "normal":
                    _add_anomaly_to_train(samples, class_label, custom_mapping.train_ratio_for_anomalies)
                elif class_mode == "anomaly":
                    if class_label != "good":
                        continue  # non-"good" classes are in "test" by default
                    # discard all good training images
                    # (adding them to "test" would create an imbalance with other anomaly classes which have fewer images)
                    samples = samples[(samples.split != "train") | (samples.label != "good")]
                    # mark the remaining test images as anomalous
                    samples.loc[samples.label == "good", "label_index"] = 1
                else:
                    raise Exception(f"unknown mode {class_mode} (must be either 'normal', 'anomaly' or 'ignore'")
    else:
        if len(categories) > 1:
            raise NotImplementedError
        category = categories[0]
        label_mapping = {}
        i = 1

        for label in set(samples.label):
            if custom_mapping.custom_labels[category].get(label) == "ignore":
                samples = samples[samples.label != label]
            elif label == "good":
                samples.loc[(samples.label == "good"), "label_index"] = 0
                label_mapping = {0: "good", **label_mapping}  # always make good '0' and always keep it as first entry
            else:
                _add_anomaly_to_train(samples, label, custom_mapping.train_ratio_for_anomalies if custom_mapping else 0.5)
                samples.loc[(samples.label == label), "label_index"] = i
                label_mapping[i] = label
                i += 1

    samples.label_index = samples.label_index.astype(int)

    if create_validation_set:
        if len(categories) > 1:
            raise NotImplementedError
        samples = create_validation_set_from_test_set(samples, seed=seed)

    # Get the data frame for the split.
    if split is not None and split in ["train", "val", "test"]:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)

    return samples, label_mapping

def _apply_custom_mapping_v2(samples, categories, custom_mapping, binary_label_indices: bool):
    """new mapping format"""
    for category in categories:
        for label in set(samples[samples.category == category].label):
            if label not in custom_mapping.custom_labels[category]:
                # remove label
                samples = samples[(samples.category != category) | (samples.label != label)]
                continue

            class_properties = custom_mapping.custom_labels[category][label]
            if "split" in class_properties:
                split = class_properties["split"]
                if isinstance(split, (list, tuple, ListConfig)):
                    assert len(split) == 2 and "train" in split and "test" in split, f"unknown {split=}"
                    if label != "good":  # "good" is already split into training and test data
                        _add_anomaly_to_train(samples, label, custom_mapping.train_ratio_for_anomalies, category)
                else:
                    if label == "good":
                        # discard entries from other split (to retain dataset somewhat balanced)
                        samples = samples[(samples.category != category) | (samples.label != label) | (samples.split == split)]
                    else:
                        # put all entries into given split
                        samples.loc[(samples.category == category) & (samples.label == label), "split"] = split

            if "label" in class_properties:  # overwrite label name
                #label_index = 0 if class_properties["label"] in ["good", "normal"] else 1
                #samples.loc[(samples.category == category) & (samples.label == label), "label_index"] = label_index
                samples.loc[(samples.category == category) & (samples.label == label), "label"] = class_properties["label"]

    if binary_label_indices:
        # Create label index for normal (0) and anomalous (1) images.
        samples.loc[:, "label_index"] = 1
        samples.loc[(samples.label == "good") | (samples.label == "normal"), "label_index"] = 0
        label_mapping = {0: "normal", 1: "anomalous"}
    else:
        label_mapping = {}
        i = 1

        for label in set(samples.label):
            if label in ["good", "normal"]:
                samples.loc[(samples.label == label), "label_index"] = 0
                label_mapping = {0: "good", **label_mapping}  # always make good '0' and always keep it as first entry
            else:
                samples.loc[(samples.label == label), "label_index"] = i
                label_mapping[i] = label
                i += 1

    return samples, label_mapping

def _add_anomaly_to_train(samples, class_label, train_ratio_for_anomalies, category=None):
    if class_label == "good":
        return  # "good" is in "train" by default

    if category:
        indices = ((samples.category == category) & (samples.label == class_label))
    else:
        indices = (samples.label == class_label)
    # note: changing `samples.label_index` is enough, we do not need to change `samples.label`
    samples.loc[indices, "label_index"] = 0

    # add some images to the training set
    num_training_samples = int(len(samples.label[indices]) * train_ratio_for_anomalies)
    samples_labeled_splits = samples.split[indices]
    samples_labeled_splits[:num_training_samples] = "train"
    samples.loc[indices, "split"] = samples_labeled_splits


def _parse_limit(limit: Union[int, float, Literal['max_anomaly', 'sum_anomalies']], units_per_class: Union[np.ndarray, Tensor]) -> int:
    if limit == "max_anomaly":
        # limit the number of good patches to the number of patches of the most common anomaly type
        return max(units_per_class[1:])
    elif limit == "sum_anomalies":
        # limit the number of good patches to the sum of all anomaly patches
        return sum(units_per_class[1:])
    elif isinstance(limit, int):
        # limit the number of good patches to a specific constant
        return limit
    elif isinstance(limit, float):
        # treat as ratio of good images
        return int(units_per_class[0] * limit)
    else:
        raise ValueError(f"unknown value for {limit=}")


class MVTecDataset(VisionDataset):
    """MVTec AD PyTorch Dataset."""

    def __init__(
        self,
        root: Union[Path, str],
        categories: Union[str, List[str]],
        pre_process: PreProcessor,
        split: str,
        task: str = "segmentation",
        seed: Optional[int] = None,
        create_validation_set: bool = False,
        custom_mapping: Optional[DictConfig] = None,
    ) -> None:
        """Mvtec AD Dataset class.

        Args:
            root: Path to the MVTec AD dataset
            category: Name of the MVTec AD category.
            pre_process: List of pre_processing object containing albumentation compose.
            split: 'train', 'val' or 'test'
            task: ``classification`` or ``segmentation``
            seed: seed used for the random subset splitting
            create_validation_set: Create a validation subset in addition to the train and test subsets

        Examples:
            >>> from anomalib.data.mvtec import MVTecDataset
            >>> from anomalib.data.transforms import PreProcessor
            >>> pre_process = PreProcessor(image_size=256)
            >>> dataset = MVTecDataset(
            ...     root='./datasets/MVTec',
            ...     category='leather',
            ...     pre_process=pre_process,
            ...     task="classification",
            ...     is_train=True,
            ... )
            >>> dataset[0].keys()
            dict_keys(['image'])

            >>> dataset.split = "test"
            >>> dataset[0].keys()
            dict_keys(['image', 'image_path', 'label'])

            >>> dataset.task = "segmentation"
            >>> dataset.split = "train"
            >>> dataset[0].keys()
            dict_keys(['image'])

            >>> dataset.split = "test"
            >>> dataset[0].keys()
            dict_keys(['image_path', 'label', 'mask_path', 'image', 'mask'])

            >>> dataset[0]["image"].shape, dataset[0]["mask"].shape
            (torch.Size([3, 256, 256]), torch.Size([256, 256]))
        """
        super().__init__(root)

        if seed is None:
            warnings.warn(
                "seed is None."
                " When seed is not set, images from the normal directory are split between training and test dir."
                " This will lead to inconsistency between runs."
            )

        if isinstance(categories, str):
            categories = [categories]

        for category in categories:
            if custom_mapping and custom_mapping.custom_labels.get(category) is None:
                custom_mapping.custom_labels[category] = {}  # prevents multiple checks for `None` down the road
            assert (not custom_mapping) \
                   or custom_mapping.custom_labels[category].get("good") != "anomaly" \
                   or task == "classification", \
                   "cannot run task 'segmentation' if 'good' is considered anomalous, due to missing ground truth"

        self.root = Path(root) if isinstance(root, str) else root
        self.split = split
        self.task = task
        self.custom_mapping = custom_mapping

        self.pre_process = pre_process

        self.samples, self.label_mapping = make_mvtec_dataset(
            path=self.root,
            categories=categories,
            split=self.split,
            seed=seed,
            create_validation_set=create_validation_set,
            custom_mapping=custom_mapping,
            binary_label_indices=(task != "classification"),
        )
        self.num_classes = 1 if len(self.label_mapping) == 2 else len(self.label_mapping)
        self.num_categories = len(set(self.samples.category))
        self.images_per_class = self._calculate_images_per_class()
        logger.warning(f"number of images for {split}: {self.images_per_class} (from categories {sorted(set(self.samples.category))})")

    def _calculate_images_per_class(self):
        if self.task == "classification":
            return np.array([sum(self.samples.label == label) for label in self.label_mapping.values()])
        else:
            return np.array([sum(self.samples.label_index == i) for i in set(self.samples.label_index)])

    def limit_patches(self):
        if self.custom_mapping and self.custom_mapping.limit_train_images is not False:
            limit = _parse_limit(self.custom_mapping.limit_train_images, self.images_per_class)
            self.samples = self.samples.groupby("label").head(limit).reset_index(drop=True)
            self.images_per_class = np.array([sum(self.samples.label == label) for label in self.label_mapping.values()])
            logger.warning(f"limited number of images for {self.split}: {self.images_per_class} (from categories {sorted(set(self.samples.category))})")

    def __len__(self) -> int:
        """Get length of the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Union[str, Tensor]]:
        """Get dataset item for the index ``index``.

        Args:
            index (int): Index to get the item.

        Returns:
            Union[Dict[str, Tensor], Dict[str, Union[str, Tensor]]]: Dict of image tensor during training.
                Otherwise, Dict containing image path, target path, image tensor, label and transformed bounding box.
        """
        if index >= len(self):
            raise IndexError


        image_path = self.samples.image_path[index]
        image = read_image(image_path)

        pre_processed_no_normalization, pre_processed = self.pre_process(image=image, also_get_without_normalization=True)

        label_index = self.samples.label_index[index]
        mask_path = self.samples.mask_path[index]

        # Only anomalous (1) images have masks in MVTec AD dataset.
        # Therefore, create empty masks for normal (0) images.
        if label_index == 0:
            mask = np.zeros(shape=image.shape[:2])
        else:
            mask = cv2.imread(mask_path, flags=0) / 255.0

        pre_processed_no_normalization, pre_processed = self.pre_process(image=image, mask=mask, also_get_without_normalization=True)

        item: Dict[str, Union[str, Tensor]] = {
            "image": pre_processed["image"],
            "image_visualization": pre_processed_no_normalization["image"],
            "images_per_class": self.images_per_class,
            "image_path": image_path,
            "label": label_index,
            "label_name": self.samples.label[index],
            "original_label_name": self.samples.original_label[index],
            "category": self.samples.category[index],
            "mask": pre_processed["mask"],
            "mask_path": mask_path,
        }

        return item


class PatchwiseWrapper:
    def __init__(
            self,
            dataset: MVTecDataset,
            embedding_extractor: EmbeddingExtractor,
            num_patches_per_dimension: int,
            anomaly_threshold: float,
            use_coreset_subsampling: bool = False,
    ):
        self.dataset = dataset
        self.num_classes = dataset.num_classes
        self.num_categories = dataset.num_categories
        self.label_mapping = dataset.label_mapping
        self.embedding_extractor = embedding_extractor
        self.num_patches = num_patches_per_dimension ** 2
        self.num_patches_per_dimension = num_patches_per_dimension
        self.anomaly_threshold = anomaly_threshold
        self.use_coreset_subsampling = use_coreset_subsampling

        self.items_map = self._create_item_map(dataset)
        self.images_per_class = torch.tensor([len(self.items_map[i]) for i in self.label_mapping])

        logger.warning(f"number of patches for {dataset.split}: {self.images_per_class} (from categories {sorted(set(self.dataset.samples.category))})")

    def _create_item_map(self, dataset):
        items_map: Dict[int, List[Dict[str, Union[str, Tensor]]]] = {i: [] for i in dataset.label_mapping}

        for item in dataset:
            embedding: Tensor = self.embedding_extractor(item["image"].unsqueeze(0))
            batch_size, num_features, width, height = embedding.shape

            assert batch_size == 1 and width == height == self.num_patches_per_dimension

            embedding = embedding.permute(0, 2, 3, 1).reshape(width * height, num_features)
            continuous_mask = cv2.resize(item["mask"].numpy(), dsize=(height, width), interpolation=cv2.INTER_AREA)

            for j in range(self.num_patches):
                x, y = divmod(j, self.num_patches_per_dimension)

                if item["label"] != 0 and continuous_mask[x, y] < self.anomaly_threshold:
                    continue

                subitem = {k: item[k] for k in ["category", "label", "label_name", "original_label_name"]}
                subitem["patch_anomaly_value"] = continuous_mask[x, y]
                subitem["image"] = embedding[j]
                items_map[subitem["label"]].append(subitem)

        return items_map

    def limit_patches(self):
        limit_train_images = self.dataset.custom_mapping.limit_train_images
        if limit_train_images is False:
            return

        limit = _parse_limit(limit_train_images, self.images_per_class)

        for label, values in self.items_map.items():
            if len(values) <= limit:
                continue

            if self.use_coreset_subsampling:
                sampler = KCenterGreedyBulk(sampling_ratio=min(limit / len(values), 1.))
                sampler.update(torch.stack([batch["image"] for batch in values]).cuda())
                self.items_map[label] = [values[i] for i in sampler.select_coreset_idxs()]
            else:
                self.items_map[label] = values[:limit]
            self.images_per_class[label] = len(self.items_map[label])

        logger.warning(f"limited number of patches for {self.dataset.split}: {self.images_per_class} (from categories {sorted(set(self.dataset.samples.category))})")

    def __len__(self):
        return sum(self.images_per_class)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError

        for i in self.label_mapping:
            if index < self.images_per_class[i]:
                item = self.items_map[i][index].copy()
                item["images_per_class"] = self.images_per_class
                return item
            index -= self.images_per_class[i]
        else:
            raise Exception("should've already risen an IndexError => implementation error?")


@DATAMODULE_REGISTRY
class MVTec(LightningDataModule):
    """MVTec AD Lightning Data Module."""

    def __init__(
        self,
        root: str,
        category: Union[str, List[str]],
        # TODO: Remove default values. IAAALD-211
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        train_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 8,
        task: str = "segmentation",
        transform_config_train: Optional[Union[str, A.Compose]] = None,
        transform_config_val: Optional[Union[str, A.Compose]] = None,
        seed: Optional[int] = None,
        create_validation_set: bool = False,
        custom_mapping: Optional[DictConfig] = None,
    ) -> None:
        """Mvtec AD Lightning Data Module.

        Args:
            root: Path to the MVTec AD dataset
            category: Name of the MVTec AD category.
            image_size: Variable to which image is resized.
            train_batch_size: Training batch size.
            test_batch_size: Testing batch size.
            num_workers: Number of workers.
            task: ``classification`` or ``segmentation``
            transform_config_train: Config for pre-processing during training.
            transform_config_val: Config for pre-processing during validation.
            seed: seed used for the random subset splitting
            create_validation_set: Create a validation subset in addition to the train and test subsets

        Examples:
            >>> from anomalib.data import MVTec
            >>> datamodule = MVTec(
            ...     root="./datasets/MVTec",
            ...     category="leather",
            ...     image_size=256,
            ...     train_batch_size=32,
            ...     test_batch_size=32,
            ...     num_workers=8,
            ...     transform_config_train=None,
            ...     transform_config_val=None,
            ... )
            >>> datamodule.setup()

            >>> i, data = next(enumerate(datamodule.train_dataloader()))
            >>> data.keys()
            dict_keys(['image'])
            >>> data["image"].shape
            torch.Size([32, 3, 256, 256])

            >>> i, data = next(enumerate(datamodule.val_dataloader()))
            >>> data.keys()
            dict_keys(['image_path', 'label', 'mask_path', 'image', 'mask'])
            >>> data["image"].shape, data["mask"].shape
            (torch.Size([32, 3, 256, 256]), torch.Size([32, 256, 256]))
        """
        super().__init__()

        self.root = root if isinstance(root, Path) else Path(root)
        self.category = category
        self.transform_config_train = transform_config_train
        self.transform_config_val = transform_config_val
        self.image_size = image_size
        self.custom_mapping = custom_mapping

        if self.transform_config_train is not None and self.transform_config_val is None:
            self.transform_config_val = self.transform_config_train

        self.pre_process_train = PreProcessor(config=self.transform_config_train, image_size=self.image_size)
        self.pre_process_val = PreProcessor(config=self.transform_config_val, image_size=self.image_size)

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        self.create_validation_set = create_validation_set
        self.task = task
        self.seed = seed

        self._train_data: Optional[Dataset] = None
        self._val_data: Optional[Dataset] = None
        self._test_data: Optional[MVTecDataset] = None
        if create_validation_set:
            self.val_data: MVTecDataset
        self.inference_data: Dataset
        self.label_mapping: Dict[int, str]

    def prepare_data(self) -> None:
        """Download the dataset if not available."""
        if self.root.is_dir():
            logger.info("Found the dataset.")
        else:
            self.root.mkdir(parents=True, exist_ok=True)

            logger.info("Downloading the Mvtec AD dataset.")
            url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094"
            dataset_name = "mvtec_anomaly_detection.tar.xz"
            zip_filename = self.root / dataset_name
            with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="MVTec AD") as progress_bar:
                urlretrieve(
                    url=f"{url}/{dataset_name}",
                    filename=zip_filename,
                    reporthook=progress_bar.update_to,
                )
            logger.info("Checking hash")
            hash_check(zip_filename, "eefca59f2cede9c3fc5b6befbfec275e")

            logger.info("Extracting the dataset.")
            with tarfile.open(zip_filename) as tar_file:
                tar_file.extractall(self.root)

            logger.info("Cleaning the tar file")
            (zip_filename).unlink()

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup train, validation and test data.

        Args:
          stage: Optional[str]:  Train/Val/Test stages. (Default value = None)

        """
        logger.info("Setting up train, validation, test and prediction datasets.")
        if stage in (None, "fit"):
            self.train_data  # create dataset
            self.train_data.limit_patches()

        self.val_data
        self.test_data
        self.label_mapping = self.test_data.label_mapping

        if stage == "predict":
            self.inference_data = InferenceDataset(
                path=self.root, image_size=self.image_size, transform_config=self.transform_config_val
            )

    def _create_dataset(self, pre_process: PreProcessor, split: str) -> MVTecDataset:
        return MVTecDataset(
            root=self.root,
            categories=self.category,
            pre_process=pre_process,
            split=split,
            task=self.task,
            seed=self.seed,
            create_validation_set=self.create_validation_set,
            custom_mapping=self.custom_mapping,
        )

    @property
    def train_data(self) -> Dataset:
        if self._train_data is None:
            self._train_data = self._create_dataset(self.pre_process_train, "train")
        return self._train_data

    @property
    def val_data(self) -> Dataset:
        if self._val_data is None:
            if self.create_validation_set:
                self._val_data = self._create_dataset(self.pre_process_val, "val")
            else:
                self._val_data = self._create_dataset(self.pre_process_val, "test")
        return self._val_data

    @property
    def test_data(self) -> MVTecDataset:
        if self._test_data is None:
            self._test_data = self._create_dataset(self.pre_process_val, "test")
        return self._test_data

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Get train dataloader."""
        return DataLoader(self.train_data, shuffle=True, batch_size=self.train_batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Get validation dataloader."""
        return DataLoader(self.val_data, shuffle=False, batch_size=self.test_batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Get test dataloader."""
        return DataLoader(self.test_data, shuffle=False, batch_size=self.test_batch_size, num_workers=self.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        """Get predict dataloader."""
        return DataLoader(
            self.inference_data, shuffle=False, batch_size=self.test_batch_size, num_workers=self.num_workers
        )

    @property
    def num_classes(self) -> int:
        return self.test_data.num_classes

    @property
    def num_categories(self) -> int:
        return self.test_data.num_categories

    def set_train_embedding_extractor(self, embedding_extractor: EmbeddingExtractor, num_patches_per_dimension: int,
                                      anomaly_threshold: float, use_coreset_subsampling: bool = False):
        self._train_data = PatchwiseWrapper(self.train_data, embedding_extractor, num_patches_per_dimension,
                                            anomaly_threshold, use_coreset_subsampling)
        self._val_data = PatchwiseWrapper(self.val_data, embedding_extractor, num_patches_per_dimension,
                                          anomaly_threshold, use_coreset_subsampling)
