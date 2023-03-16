from typing import Tuple

import cv2
import numpy as np
from torch import Tensor
from jaxtyping import Bool, Float


def separate_anomaly_embeddings(
        embedding: Float[Tensor, "f p p"],
        ground_truth: Float[Tensor, "w h"],
        anomaly_threshold: float
) -> Tuple[Float[Tensor, "_ f"], Float[Tensor, "_ f"]]:
    """Split a single `embedding` into normal and anomalous based on `ground_truth` and `anomaly_threshold`.

    Args:
        embedding (Tensor): embeddings to be split into normal and anomalous; shape: [num_features, width, height]
        ground_truth (Tensor): ground truth containing 1 for anomaly and 0 for normal patches; original image size
        anomaly_threshold (float): part of a patch that must be labeled anomalous in order to classify it as such

    Returns:
        Tuple[Tensor, Tensor]: two tensors with the embeddings where rescaled ground truth is below/above the
            threshold, respectively; shape: [num_embeddings, num_features]
    """
    embedding: Float[Tensor, "p p f"] = embedding.permute(1, 2, 0)  # make channel-last
    ground_truth: Float[np.ndarray, "p p"] = cv2.resize(ground_truth.cpu().numpy(), dsize=embedding.shape[:2], interpolation=cv2.INTER_AREA)
    anomalous_indices: Bool[np.ndarray, "p p"] = ground_truth >= anomaly_threshold
    anomalous_embedding: Float[Tensor, "_ f"] = embedding[anomalous_indices, :]
    normal_embedding: Float[Tensor, "_ f"] = embedding[~anomalous_indices, :]
    return normal_embedding, anomalous_embedding
