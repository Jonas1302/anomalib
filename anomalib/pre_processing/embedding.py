from typing import Tuple

import cv2
from torch import Tensor


def separate_anomaly_embeddings(embedding: Tensor, ground_truth: Tensor, anomaly_threshold: float) -> Tuple[Tensor, Tensor]:
    """Split a single `embedding` into normal and anomalous based on `ground_truth` and `self.anomaly_threshold`.

    Args:
        embedding (Tensor): embeddings to be split into normal and anomalous; shape: [num_features, width, height]
        ground_truth (Tensor): ground truth containing 1 for anomaly and 0 for normal patches; original image size
        anomaly_threshold (float): part of a patch that must be labeled anomalous in order to classify it as such

    Returns:
        Tuple[Tensor, Tensor]: two tensors with the embeddings where rescaled ground truth is below/above the
            threshold, respectively; shape: [num_embeddings, num_features]
    """
    embedding = embedding.permute(1, 2, 0)  # make channel-last
    ground_truth = cv2.resize(ground_truth.cpu().numpy(), dsize=embedding.shape[:2], interpolation=cv2.INTER_AREA)
    anomalous_indices = ground_truth >= anomaly_threshold
    anomalous_embedding = embedding[anomalous_indices, :]
    normal_embedding = embedding[~anomalous_indices, :]
    return normal_embedding, anomalous_embedding
