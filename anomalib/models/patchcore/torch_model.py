"""PyTorch model for the PatchCore model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Dict, List, Optional, Tuple, Union

from jaxtyping import Bool, Float, Int
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from anomalib.models.components import DynamicBufferModule
from anomalib.models.components.feature_extractors.timm import get_feature_extractor
from anomalib.models.components.sampling.k_center_greedy import KCenter
from anomalib.models.patchcore.anomaly_map import AnomalyMapGenerator
from anomalib.pre_processing import Tiler
from anomalib.pre_processing.embedding import separate_anomaly_embeddings


class PatchcoreModel(DynamicBufferModule, nn.Module):
    """Patchcore Module."""

    def __init__(
        self,
        input_size: Tuple[int, int],
        layers: List[str],
        backbone: str = "wide_resnet50_2",
        pre_trained: bool = True,
        num_neighbors: int = 9,
        anomaly_map_with_neighbours: bool = False,
        locally_aware_patch_features: bool = True,
        coreset_sampling: KCenter = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.tiler: Optional[Tiler] = None

        self.backbone = backbone
        self.layers = layers
        self.input_size = input_size
        self.num_neighbors = num_neighbors
        self.anomaly_map_with_neighbours = anomaly_map_with_neighbours
        self.locally_aware_patch_features = locally_aware_patch_features
        self.coreset_sampling = coreset_sampling

        self.feature_extractor = get_feature_extractor(self.backbone, pre_trained=pre_trained, layers=self.layers)
        self.feature_extractor.eval()
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)

        self.register_buffer("memory_bank", Tensor())
        self.memory_bank: Tensor

    def forward(self, input_tensor: Float[Tensor, "b c w h"], **kwargs) \
            -> Optional[Tuple[Tensor, Tensor]]:
        """Return Embedding during training, or a tuple of anomaly map and anomaly score during testing.

        Steps performed:
        1. Get features from a CNN.
        2. Generate embedding based on the features.
        3. Compute anomaly map in test mode.

        Args:
            input_tensor (Tensor): Input tensor

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: Embedding for training,
                anomaly map and anomaly score for testing.
        """
        embedding = self.get_embedding(input_tensor)

        batch_size, _, width, height = embedding.shape
        embedding = self.reshape_embedding(embedding)

        if self.training:
            self.coreset_sampling.update(embedding)
        else:
            # apply nearest neighbor search
            patch_scores, locations = self.nearest_neighbors(embedding=embedding, n_neighbors=1,
                                                             memory_bank=self.memory_bank)
            # reshape to batch dimension
            patch_scores = patch_scores.reshape((batch_size, -1))
            locations = locations.reshape((batch_size, -1))
            if self.anomaly_map_with_neighbours:
                # compute score for each pixel with its neighbours
                patch_scores = self.compute_anomaly_score_map(patch_scores, locations, embedding, self.memory_bank)
                anomaly_score, _ = patch_scores.max(dim=1)
            else:
                # compute anomaly score
                anomaly_score = self.compute_anomaly_score(patch_scores, locations, embedding, self.memory_bank)
            # reshape to w, h
            patch_scores = patch_scores.reshape((batch_size, 1, width, height))
            # get anomaly map
            anomaly_map = self.anomaly_map_generator(patch_scores)
            return anomaly_map, anomaly_score

    def get_embedding(self, input_tensor: Float[Tensor, "b _ w h"]) -> Float[Tensor, "b f p p"]:
        """Extracts the features from the feature_extractor and creates the embedding."""
        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        # extract the features
        with torch.no_grad():
            features: Dict[str, Float[Tensor, "b _ _ _"]] = self.feature_extractor(input_tensor)

        # filter and pool the features
        if self.locally_aware_patch_features in [True, "true"]:
            features = {layer: self.feature_pooler(feature) for layer, feature in features.items()}
        elif self.locally_aware_patch_features == "both":
            features = {layer: torch.cat((feature, self.feature_pooler(feature)), dim=1)
                        for layer, feature in features.items()}

        # create embeddings from the features
        embedding: Float[Tensor, "b _ p p"] = features[str(self.layers[0])]
        for layer in self.layers[1:]:
            layer_embedding = features[str(layer)]
            layer_embedding = F.interpolate(layer_embedding, size=embedding.shape[-2:], mode="nearest")
            embedding = torch.cat((embedding, layer_embedding), 1)

        if self.tiler:
            embedding = self.tiler.untile(embedding)

        return embedding

    def calculate_coreset(self):
        """Creates a subsampled coreset from the collected embeddings."""
        self.memory_bank = self.coreset_sampling.get_coreset()

    @staticmethod
    def reshape_embedding(embedding: Float[Tensor, "b f p p"]) -> Float[Tensor, "b*p*p f"]:
        """Reshape Embedding.

        Reshapes Embedding to the following format:
        [Batch, Embedding, Patch, Patch] to [Batch*Patch*Patch, Embedding]

        Args:
            embedding (Tensor): Embedding tensor extracted from CNN features.

        Returns:
            Tensor: Reshaped embedding tensor.
        """
        embedding_size = embedding.size(1)
        embedding = embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)
        return embedding

    @staticmethod
    def nearest_neighbors(embedding: Float[Tensor, "b*p*p f"], n_neighbors: int, memory_bank: Tensor) -> Tuple[Tensor, Tensor]:
        """Nearest Neighbours using brute force method and euclidean norm.

        Args:
            embedding (Tensor): Features to compare the distance with the memory bank.
            n_neighbors (int): Number of neighbors to look at

        Returns:
            Tensor: Patch scores.
            Tensor: Locations of the nearest neighbor(s).
        """
        distances = torch.cdist(embedding, memory_bank, p=2.0)  # euclidean norm
        if n_neighbors == 1:
            # when n_neighbors is 1, speed up computation by using min instead of topk
            patch_scores, locations = distances.min(1)
        else:
            patch_scores, locations = distances.topk(k=n_neighbors, largest=False, dim=1)
        return patch_scores, locations

    def compute_anomaly_score(self, patch_scores: Tensor, locations: Tensor, embedding: Tensor,
                              memory_bank: Tensor, mode=torch.argmax) -> Tensor:
        """Compute Image-Level Anomaly Score.

        Args:
            patch_scores (Tensor): Patch-level anomaly scores
            locations: Memory bank locations of the nearest neighbor for each patch location
            embedding: The feature embeddings that generated the patch scores
        Returns:
            Tensor: Image-level anomaly scores
        """
        # Don't need to compute weights if num_neighbors is 1
        if self.num_neighbors == 1:
            return patch_scores.amax(1)
        batch_size, num_patches = patch_scores.shape
        # 1. Find the patch with the largest distance to it's nearest neighbor in each image
        max_patches_idx = mode(patch_scores, dim=1)  # indices of m^test,* in the paper
        # m^test,* in the paper
        max_patches_features = embedding.reshape(batch_size, num_patches, -1)[torch.arange(batch_size), max_patches_idx]
        # 2. Find the distance of the patch to it's nearest neighbor, and the location of the nn in the membank
        score = patch_scores[torch.arange(batch_size), max_patches_idx]  # s^* in the paper
        nn_index = locations[torch.arange(batch_size), max_patches_idx]  # indices of m^* in the paper
        # 3. Find the support samples of the nearest neighbor in the membank
        nn_sample = memory_bank[nn_index, :]  # m^* in the paper
        # indices of N_b(m^*) in the paper
        _, support_samples = self.nearest_neighbors(nn_sample, self.num_neighbors, memory_bank)
        # 4. Find the distance of the patch features to each of the support samples
        distances = torch.cdist(max_patches_features.unsqueeze(1), memory_bank[support_samples], p=2.0)
        # 5. Apply softmax to find the weights
        weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]
        # 6. Apply the weight factor to the score
        score = weights * score  # s in the paper
        return score

    def compute_anomaly_score_map(self, patch_scores: Float[Tensor, "b p*p"], locations: Float[Tensor, "b p*p"],
                                  embedding: Float[Tensor, "p*p f"], memory_bank: Tensor) -> Float[Tensor, "b p*p"]:
        """Compute the anomaly score for each patch."""
        # Don't need to compute weights if num_neighbors is 1
        if self.num_neighbors == 1:
            return patch_scores
        batch_size, num_patches = patch_scores.shape
        patch_scores = patch_scores.reshape(batch_size * num_patches, 1)  # s^* in the paper
        # 2. Find the distance of the patch to it's nearest neighbor, and the location of the nn in the membank
        locations = locations.reshape(batch_size * num_patches)  # indices of m^* in the paper
        # 3. Find the support samples of the nearest neighbor in the membank
        nn_sample = memory_bank[locations, :]  # m^* in the paper
        # indices of N_b(m^*) in the paper
        _, support_samples = self.nearest_neighbors(nn_sample, self.num_neighbors, memory_bank)
        # 4. Find the distance of the patch features to each of the support samples
        distances = torch.cdist(embedding.unsqueeze(1), memory_bank[support_samples], p=2.0)
        # 5. Apply softmax to find the weights
        weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]
        # 6. Apply the weight factor to the scores
        scores = weights * patch_scores.squeeze(1)  # s in the paper
        return scores.reshape(batch_size, num_patches)


class LabeledPatchcore(PatchcoreModel):
    """A Patchcore model which stores multiple coresets/memory banks.

    This model can classify images and their patches by using multiple coresets as well as return an anomaly score
    per anomaly type for each patch.
    """

    def __init__(self, *args, anomaly_threshold: float, num_classes: int, **kwargs):
        super().__init__(*args, **kwargs)
        del self.memory_bank
        self.anomaly_threshold = anomaly_threshold
        self.num_classes = num_classes
        self.memory_banks = LabeledPatchcore.MemoryBanks(self)
        self.coreset_sampling.min_num_embeddings = 1000  # use at least 1000 embeddings from each class (if possible)
        self.coreset_samplings: Dict[str, KCenter] = {}  # will contain one entry per label

    def forward(self, input_tensor: Float[Tensor, "b _ w h"], ground_truths: Optional[Float[Tensor, "b w h"]] = None,
                labels: Optional[Int[Tensor, "b"]] = None) \
            -> Union[None,
                     Tuple[Float[Tensor, "b c w h"], Float[Tensor, "b c"]],
                     Tuple[Float[Tensor, "b c w h"], Float[Tensor, "b c"], Float[Tensor, "b c p p"]]]:
        embeddings: Float[Tensor, "b f p p"] = self.get_embedding(input_tensor)
        batch_size, _, width, height = embeddings.shape

        if self.training:
            for i in range(batch_size):
                embedding: Float[Tensor, "_ f"]
                if labels[i] != 0:
                    # only add anomalous embeddings
                    _, embedding = separate_anomaly_embeddings(embeddings[i], ground_truths[i], self.anomaly_threshold)
                else:
                    embedding = self.reshape_embedding(embeddings[i].unsqueeze(0))
                # Note: `self.coreset_sampling` is used as an empty blueprint, it allows us to share its projection
                #   model across all samplings by copying it
                self.coreset_samplings.setdefault(labels[i].item(), copy.copy(self.coreset_sampling)).update(embedding)
                assert 0 <= labels[i] < self.num_classes, f"{labels[i]=}"
            return
        else:
            results = []
            for i in range(batch_size):
                embedding: Float[Tensor, "p*p f"] = self.reshape_embedding(embeddings[i].unsqueeze(0))
                results.append(self._forward_single(embedding, width, height))

            return tuple(map(torch.cat, zip(*results)))

    def _forward_single(self, embedding: Float[Tensor, "p*p f"], width: int, height: int) \
            -> Union[Tuple[Float[Tensor, "1 c w h"], Float[Tensor, "1 c"]],
                     Tuple[Float[Tensor, "1 c w h"], Float[Tensor, "1 c"], Float[Tensor, "1 c p p"]]]:
        """Forwards an embedding with batch size 1.

        Args:
            embedding (Tensor): embedding with batch size 1
            width, height: number of patches per row and column
        Returns:
            Tuple[Tensor, Tensor, Tensor]: anomaly maps (one per label), anomaly score (one per label) as normalized
                score in [0, 1] for the predicted label and zero for all others and the unscaled anomaly map with one
                entry per patch; all with batch-size 1
        """
        device = embedding.device
        anomaly_patch_maps: Dict[int, Float[Tensor, "1 p*p"]] = {}

        # collect the anomaly scores for each patch and label
        for i in range(self.num_classes):
            # apply nearest neighbor search
            patch_scores, locations = self.nearest_neighbors(embedding=embedding, n_neighbors=1, memory_bank=self.memory_banks[i])
            # reshape to batch dimension
            patch_scores: Float[Tensor, "1 p*p"] = patch_scores.reshape((1, -1))
            locations: Float[Tensor, "1 p*p"] = locations.reshape((1, -1))
            # compute weighted anomaly score for each patch
            scores: Float[Tensor, "1 p*p"] = self.compute_anomaly_score_map(patch_scores, locations, embedding, self.memory_banks[i])
            anomaly_patch_maps[i] = scores

        # lowest anomaly scores for each patch
        closest_anomaly_map: Float[Tensor, "p*p"] = torch.cat(list(anomaly_patch_maps.values())[1:]).min(dim=0)[0]
        # calculate anomaly map per label; values are between 0 and 1 where a higher value means closer to i-th coreset
        normed_anomaly_patch_maps: Float[Tensor, "c 1 p*p"] = torch.stack([
            self._get_anomaly_map(closest_anomaly_map, anomaly_patch_maps[0]) if i == 0 else
            self._get_anomaly_map(anomaly_patch_maps[0], anomaly_patch_maps[i])
            for i in range(self.num_classes)
        ])
        normed_anomaly_patch_maps: Float[Tensor, "c 1 p p"] = normed_anomaly_patch_maps.reshape((self.num_classes, 1, width, height))

        normed_anomaly_maps: Float[Tensor, "1 c w h"] = self.anomaly_map_generator(normed_anomaly_patch_maps).permute(1, 0, 2, 3)
        normed_anomaly_patch_maps: Float[Tensor, "1 c p p"] = normed_anomaly_patch_maps.permute(1, 0, 2, 3)
        return normed_anomaly_maps.to(device), normed_anomaly_patch_maps.to(device)

    @staticmethod
    def _get_anomaly_map(reference_map: Tensor, anomaly_map: Tensor) -> Tensor:
        """Return a normed (values in [0, 1]) anomaly map, whereas a higher values means closer to `anomaly_map`."""
        return reference_map / (reference_map + anomaly_map)

    def calculate_coreset(self):
        for label, coreset_sampling in self.coreset_samplings.items():
            self.memory_banks[label] = coreset_sampling.get_coreset()

    class MemoryBanks:
        """Creates and allows access to memory banks.

        The memory banks are registered as buffers of the given model, which (should) mean that they are properly
        saved and loaded with the model.
        """

        def __init__(self, model):
            self._model = model

        def __getitem__(self, key):
            return getattr(self._model, f"_memory_bank_{key}")

        def __setitem__(self, key, value):
            assert isinstance(value, Tensor)
            memory_bank_name = f"_memory_bank_{key}"
            if not hasattr(self._model, memory_bank_name):
                self._model.register_buffer(memory_bank_name, value)
            else:
                setattr(self._model, memory_bank_name, value)
