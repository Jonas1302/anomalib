"""This module comprises PatchCore Sampling Methods for the embedding.

- k Center Greedy Method
    Returns points that minimizes the maximum distance of any point to a center.
    . https://arxiv.org/abs/1708.00489
"""
from abc import ABC, abstractmethod
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from anomalib.models.components.dimensionality_reduction import SparseRandomProjection


class KCenter(ABC):
    def __init__(self, sampling_ratio: float, min_num_embeddings: int = 0):
        self._embedding: List[Tensor] = []
        self.sampling_ratio = sampling_ratio
        self.min_num_embeddings = min_num_embeddings
        self.model = SparseRandomProjection(eps=0.9)
        self.uptodate = True
        self.device = "cpu"

    def __copy__(self):
        copy = self.__class__(self.sampling_ratio, self.min_num_embeddings)
        copy.model = self.model
        return copy

    def update(self, embedding: Tensor):
        self.device = embedding.device
        self._embedding.append(embedding.cpu())
        self.uptodate = False

    @property
    def embedding(self):
        if len(self._embedding) == 0:
            raise Exception("no embeddings added")
        if len(self._embedding) > 1:
            self._embedding = [torch.cat(self._embedding)]
        torch.cuda.empty_cache()  # clear as much space as possible
        return self._embedding[0].to(self.device)

    @embedding.setter
    def embedding(self, embedding: Optional[Tensor]):
        if embedding is None:
            self._embedding = []
        else:
            self.device = embedding.device
            self._embedding = [embedding.cpu()]
        self.uptodate = False

    @property
    def coreset_size(self):
        num_embeddings = self.embedding.shape[0]
        return int(max(min(num_embeddings, self.min_num_embeddings), num_embeddings * self.sampling_ratio))

    def cleanup(self):
        """Release memory, may prohibit further training/subsampling"""
        assert self.uptodate
        self._embedding = None
        torch.cuda.empty_cache()

    @abstractmethod
    def get_coreset(self):
        """Returns the coreset. Some implementations may calculate the coreset here."""


class KCenterGreedyBulk(KCenter):
    """Implements k-center-greedy method.

    Args:
        sampling_ratio (float): Ratio to choose coreset size from the embedding size.
    """

    def __init__(self, sampling_ratio: float, min_num_embeddings: int = 0) -> None:
        super().__init__(sampling_ratio, min_num_embeddings)

        self.features: Tensor
        self.min_distances: Optional[Tensor] = None
        self.coreset = None

    def reset_distances(self) -> None:
        """Reset minimum distances."""
        self.min_distances = None

    def update_distances(self, cluster_centers: List[int]) -> None:
        """Update min distances given cluster centers.

        Args:
            cluster_centers (List[int]): indices of cluster centers
        """

        if cluster_centers:
            centers = self.features[cluster_centers]

            distance = F.pairwise_distance(self.features, centers, p=2).reshape(-1, 1)

            if self.min_distances is None:
                self.min_distances = distance
            else:
                self.min_distances = torch.minimum(self.min_distances, distance)

    def get_new_idx(self) -> int:
        """Get index value of a sample.

        Based on minimum distance of the cluster

        Returns:
            int: Sample index
        """

        if isinstance(self.min_distances, Tensor):
            idx = int(torch.argmax(self.min_distances).item())
        else:
            raise ValueError(f"self.min_distances must be of type Tensor. Got {type(self.min_distances)}")

        return idx

    def select_coreset_idxs(self, selected_idxs: Optional[List[int]] = None) -> List[int]:
        """Greedily form a coreset to minimize the maximum distance of a cluster.

        Args:
            selected_idxs: index of samples already selected. Defaults to an empty set.

        Returns:
          indices of samples selected to minimize distance to cluster centers
        """
        if self.coreset_size == len(self.embedding):
            return list(range(self.coreset_size))

        if selected_idxs is None:
            selected_idxs = []

        if self.embedding.ndim == 2:
            # train the reduction projection in the first run
            if not hasattr(self.model, "sparse_random_matrix") or self.model.sparse_random_matrix is None:
                self.model.fit(self.embedding)
            self.features = self.model.transform(self.embedding)
            self.reset_distances()
        else:
            self.features = self.embedding.reshape(self.embedding.shape[0], -1)
            self.update_distances(cluster_centers=selected_idxs)

        selected_coreset_idxs: List[int] = []
        idx = int(torch.randint(high=len(self.embedding), size=(1,)).item())
        for _ in range(self.coreset_size):
            self.update_distances(cluster_centers=[idx])
            idx = self.get_new_idx()
            if idx in selected_idxs:
                raise ValueError("New indices should not be in selected indices.")
            self.min_distances[idx] = 0
            selected_coreset_idxs.append(idx)

        return selected_coreset_idxs

    def get_coreset(self, selected_idxs: Optional[List[int]] = None) -> Tensor:
        """Select coreset from the embedding.

        Args:
            selected_idxs: index of samples already selected. Defaults to an empty set.

        Returns:
            Tensor: Output coreset

        Example:
            >>> embedding.shape
            torch.Size([219520, 1536])
            >>> sampler = KCenterGreedyBulk(...)
            >>> coreset = sampler.sample_coreset()
            >>> coreset.shape
            torch.Size([219, 1536])
        """
        if self.uptodate:
            return self.coreset
        if self.coreset_size == len(self.embedding):
            return self.embedding

        idxs = self.select_coreset_idxs(selected_idxs)
        self.coreset = self.embedding[idxs]
        self.uptodate = True

        return self.coreset


class KCenterGreedyOnline(KCenter):
    """Implements k-center-greedy method for incremental usage."""

    def __init__(self, sampling_ratio: float, min_num_embeddings: int = 0) -> None:
        super().__init__(sampling_ratio, min_num_embeddings)

        self.reduced_coreset: List[Tensor] = []  # coreset with the reduced embeddings after applying the projection
        self.coreset: List[Tensor] = []  # M_C in the paper

    def _calculate_min_distances(self, features: Tensor) -> Tensor:
        """Return a new tensor containing the min distances for a random center."""

        if self.reduced_coreset:  # we already have some embeddings in the coreset
            centers = torch.stack(self.reduced_coreset).to(self.device)
            min_distances = F.pairwise_distance(features, centers[0], p=2).reshape(-1, 1)
            for i in range(1, centers.shape[0]):
                min_distances = self._update_min_distances(features, min_distances, centers[i])
            return min_distances
        else:  # no embeddings in coreset yet => assume a random one is already added
            centers = features[int(torch.randint(high=features.shape[0], size=(1,)).item())]
            return F.pairwise_distance(features, centers, p=2).reshape(-1, 1)

    def _update_min_distances(self, features: Tensor, min_distances: Tensor, new_coreset_patch: Tensor) -> Tensor:
        """Return an updated tensor conatining the min distances given the new coreset patch."""
        distance = F.pairwise_distance(features, new_coreset_patch, p=2).reshape(-1, 1)
        return torch.minimum(min_distances, distance)

    def update(self, embedding: Tensor):
        """Update the coreset using the given embeddings."""

        if embedding.ndim != 2:
            raise ValueError(f"unknown dimension for embedding ({embedding.ndim})")

        self.device = embedding.device
        additional_selected_idxs = []
        additional_coreset_size = max(1, int(embedding.shape[0] * self.sampling_ratio))
        # train the reduction projection in the first run
        if not hasattr(self.model, "sparse_random_matrix") or self.model.sparse_random_matrix is None:
            self.model.fit(embedding)
        # reduce embeddings, psi in the paper
        reduced_embeddings = self.model.transform(embedding)
        # calculate the distances between the embeddings and current coreset
        min_distances = self._calculate_min_distances(reduced_embeddings)

        for _ in range(additional_coreset_size):
            idx = int(torch.argmax(min_distances).item())  # index of new sample, m_i in the paper
            assert idx not in additional_selected_idxs, "New indices should not be in selected indices."
            min_distances[idx] = 0  # ensure index is never picked again
            additional_selected_idxs.append(idx)
            self.reduced_coreset.append(reduced_embeddings[idx].cpu())
            self.coreset.append(embedding[idx].cpu())
            min_distances = self._update_min_distances(reduced_embeddings, min_distances, reduced_embeddings[idx])

    def get_coreset(self):
        return torch.stack(self.coreset).to(self.device)


class KCenterGreedyOnDemand(KCenterGreedyOnline):
    """Subsampling method that updates the coreset whenever `.get_coreset()` is called with the recent embeddings."""

    def update(self, embedding: Tensor):
        super(KCenterGreedyOnline, self).update(embedding)  # adds `embedding` to `self.embedding`

    def get_coreset(self):
        super().update(self.embedding)  # calculates all stored embeddings in one bulk
        self.embedding = None  # forget about old embeddings in case there'll be new ones
        return super().get_coreset()


class KCenterRandom(KCenter):
    """A coreset subsampling that randomly selects embeddings."""
    def get_coreset(self) -> Tensor:
        import numpy as np

        indices = np.random.choice(self.embedding.shape[0], self.coreset_size, replace=False)
        coreset = self.embedding[indices]
        return coreset

class KCenterAll(KCenter):
    """A "subsampling" that returns all embeddings as coreset.

    This is equivalent to `KCenterGreedyBulk(sampling_ratio=1.0)` but way faster because no actual subsampling happens.
    """
    def __init__(self, sampling_ratio=None, min_num_embeddings=0):
        super().__init__(sampling_ratio=1.0)

    def get_coreset(self):
        return self.embedding
