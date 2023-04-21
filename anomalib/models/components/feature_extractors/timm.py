"""Feature Extractor.

This script extracts features from a CNN network
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import abstractmethod, ABC

import math
import warnings
from typing import Dict, List, Union

import timm
import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


def get_feature_extractor(backbone_name: str, layers: List[str], pre_trained: bool = True, pretrained_model=None):
    if pretrained_model is None and not backbone_name.startswith("vit"):
        return FeatureExtractor(backbone_name, layers=layers, pre_trained=pre_trained)
    elif "resnet" in backbone_name:
        return ResNet(backbone_name, layers=layers, pre_trained=pre_trained, pretrained_backbone=pretrained_model)
    elif backbone_name.startswith("vit"):
        return ViT(backbone_name, layers=layers, pre_trained=pre_trained, pretrained_backbone=pretrained_model)
    else:
        raise ValueError(f"unknown {backbone_name=}")


class TimmFeatureExtractor(nn.Module):
    """Extract features from a CNN.

    Args:
        backbone (nn.Module): The backbone to which the feature extraction hooks are attached.
        layers (Iterable[str]): List of layer names of the backbone to which the hooks are attached.
        pre_trained (bool): Whether to use a pre-trained backbone. Defaults to True.
        requires_grad (bool): Whether to require gradients for the backbone. Defaults to False.
            Models like ``stfpm`` use the feature extractor model as a trainable network. In such cases gradient
            computation is required.

    Example:
        >>> import torch
        >>> from anomalib.models.components.feature_extractors import TimmFeatureExtractor

        >>> model = TimmFeatureExtractor(model="resnet18", layers=['layer1', 'layer2', 'layer3'])
        >>> input = torch.rand((32, 3, 256, 256))
        >>> features = model(input)

        >>> [layer for layer in features.keys()]
            ['layer1', 'layer2', 'layer3']
        >>> [feature.shape for feature in features.values()]
            [torch.Size([32, 64, 64, 64]), torch.Size([32, 128, 32, 32]), torch.Size([32, 256, 16, 16])]
    """

    def __init__(self, backbone: str, layers: List[str], pre_trained: bool = True, requires_grad: bool = False):
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self.idx = self._map_layer_to_idx()
        self.requires_grad = requires_grad
        if isinstance(backbone, str):
            self.feature_extractor = timm.create_model(
                backbone,
                pretrained=pre_trained,
                features_only=True,
                exportable=True,
                out_indices=self.idx,
            )
        else:
            self.feature_extractor = backbone
        #self.out_dims = self.feature_extractor.feature_info.channels()
        self._features = {layer: torch.empty(0) for layer in self.layers}

    def _map_layer_to_idx(self, offset: int = 3) -> List[int]:
        """Maps set of layer names to indices of model.

        Args:
            offset (int) `timm` ignores the first few layers when indexing please update offset based on need

        Returns:
            Feature map extracted from the CNN
        """
        idx = []
        if isinstance(self.backbone, str):
            features = timm.create_model(
                self.backbone,
                pretrained=False,
                features_only=False,
                exportable=True,
            )
        else:
            features = self.backbone
        for i in self.layers:
            try:
                idx.append(list(dict(features.named_children()).keys()).index(i) - offset)
            except ValueError:
                warnings.warn(f"Layer {i} not found in model {self.backbone}")
                # Remove unfound key from layer dict
                self.layers.remove(i)

        return idx

    def forward(self, inputs: Tensor) -> Dict[str, Tensor]:
        """Forward-pass input tensor into the CNN.

        Args:
            inputs (Tensor): Input tensor

        Returns:
            Feature map extracted from the CNN
        """
        if self.requires_grad:
            features = dict(zip(self.layers, self.feature_extractor(inputs)))
        else:
            self.feature_extractor.eval()
            with torch.no_grad():
                features = dict(zip(self.layers, self.feature_extractor(inputs)))
        return features


class FeatureExtractor(TimmFeatureExtractor):
    """Compatibility wrapper for the old FeatureExtractor class.

    See :class:`anomalib.models.components.feature_extractors.timm.TimmFeatureExtractor` for more details.
    """

    def __init__(self, *args, **kwargs):
        logger.warning(
            "FeatureExtractor is deprecated. Use TimmFeatureExtractor instead."
            " Both FeatureExtractor and TimmFeatureExtractor will be removed in version 2023.1"
        )
        super().__init__(*args, **kwargs)


class FeatureExtractor2(nn.Module, ABC):
    def __init__(self, backbone_name: str = "vit_base_patch16_224", *, layers: Union[List[str], List[int]], pre_trained=True, pretrained_backbone=None):
        super().__init__()

        if pretrained_backbone:
            self.model = pretrained_backbone
        else:
            self.model = timm.create_model(backbone_name, pretrained=pre_trained)
        self.model.eval()

        # from https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6
        self.activation = {}
        blocks = self.get_blocks()

        for layer in layers:
            blocks[str(layer)].register_forward_hook(self.get_activation(str(layer)))

    def forward(self, in_tensor):
        with torch.no_grad():
            self.model(in_tensor)
        return self.activation

    @abstractmethod
    def get_blocks(self):
        pass

    @abstractmethod
    def get_activation(self, name):
        pass

class ViT(FeatureExtractor2):
    def get_blocks(self):
        return dict(dict(self.model.named_children())["blocks"].named_children())

    def get_activation(self, name):
        def hook(model, input_, output):
            batch_size, num_patches, num_features = output.shape
            width = height = int(math.sqrt(num_patches - 1))
            assert width * height == num_patches - 1
            self.activation[name] = output.detach()[:, 1:, :].permute(0, 2, 1).reshape(batch_size, num_features, width, height)
        return hook

class ResNet(FeatureExtractor2):
    def get_blocks(self):
        return dict(self.model.named_children())

    def get_activation(self, name):
        def hook(model, input_, output):
            self.activation[name] = output.detach()
        return hook
