"""Implementation of AnomalyScoreThreshold based on TorchMetrics."""
from typing import List

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torchmetrics import PrecisionRecallCurve


class AnomalyScoreThreshold(PrecisionRecallCurve):
    """Anomaly Score Threshold.

    This class computes/stores the threshold that determines the anomalous label
    given anomaly scores. If the threshold method is ``manual``, the class only
    stores the manual threshold values.

    If the threshold method is ``adaptive``, the class initially computes the
    adaptive threshold to find the optimal f1_score and stores the computed
    adaptive threshold value.
    """

    def __init__(self, default_value: float = 0.5, num_classes: int = 1, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)

        self.default_value = default_value
        self.add_state("value", default=torch.full((num_classes,), self.default_value, dtype=torch.float), persistent=True)  # pylint: disable=not-callable
        self.value = torch.full((num_classes,), self.default_value, dtype=torch.float)  # pylint: disable=not-callable

    def compute(self) -> torch.Tensor:
        """Compute the threshold that yields the optimal F1 score.

        Compute the F1 scores while varying the threshold. Store the optimal
        threshold as attribute and return the maximum value of the F1 score.

        Returns:
            Value of the F1 score at the optimal threshold.
        """
        precisions: List[torch.Tensor]
        recalls: List[torch.Tensor]
        thresholds: List[torch.Tensor]

        precisions, recalls, thresholds = super().compute()
        precisions = [precisions] if not isinstance(precisions, list) else precisions
        recalls = [recalls] if not isinstance(recalls, list) else recalls
        thresholds = [thresholds] if not isinstance(thresholds, list) else thresholds

        for i, (precision, recall, threshold) in enumerate(zip(precisions, recalls, thresholds)):
            f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
            if threshold.dim() == 0:
                # special case where recall is 1.0 even for the highest threshold.
                # In this case 'thresholds' will be scalar.
                self.value[i] = threshold
            else:
                self.value[i] = threshold[torch.argmax(f1_score)]
        return self.value[0] if self.num_classes == 1 else self.value
