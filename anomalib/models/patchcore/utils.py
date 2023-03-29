from typing import Dict, List, Optional, Tuple, Union

from jaxtyping import Bool, Float
import torch
from torch import Tensor

from anomalib.utils.metrics import AnomalyScoreThreshold


def process_pred_masks(
        anomaly_maps: Float[Tensor, "b c w h"],
        anomaly_patch_maps: Float[Tensor, "b c p p"],
        outputs: Dict[str, Union[List, Tensor]],
        threshold: Optional[AnomalyScoreThreshold] = None,
) -> Tuple[Bool[Tensor, "b c w h"], Bool[Tensor, "b c p p"]]:
    """Creates prediction masks (bool) from the anomaly maps (float)."""
    if threshold is None:
        # from https://stackoverflow.com/a/72628126
        pred_masks: Bool[Tensor, "b c w h"] = torch.zeros_like(anomaly_maps, dtype=torch.bool) \
            .scatter(dim=1, index=anomaly_maps.argmax(dim=1, keepdim=True), value=True)
        pred_patch_masks: Bool[Tensor, "b c p p"] = torch.zeros_like(anomaly_patch_maps, dtype=torch.bool) \
            .scatter(dim=1, index=anomaly_patch_maps.argmax(dim=1, keepdim=True), value=True)
    elif len(threshold.value.shape) > 0:  # one threshold for each class
        # note: this means a patch can be classified as multiple classes if `sum(threshold.value)` < 1
        pred_masks: Bool[Tensor, "b w h c"] = anomaly_maps.permute(0, 2, 3, 1) >= threshold.value
        pred_masks: Bool[Tensor, "b c w h"] = pred_masks.permute(0, 3, 1, 2)  # permute back to original shape
        pred_patch_masks: Bool[Tensor, "b p p c"] = anomaly_patch_maps.permute(0, 2, 3, 1) >= threshold.value
        pred_patch_masks: Bool[Tensor, "b c p p"] = pred_patch_masks.permute(0, 3, 1, 2)
    else:  # one threshold for all classes
        pred_masks: Bool[Tensor, "b c w h"] = anomaly_maps >= threshold.value
        pred_patch_masks: Bool[Tensor, "b c p p"] = anomaly_patch_maps >= threshold.value

    outputs["pred_masks"] = pred_masks
    return pred_masks, pred_patch_masks


def process_label_and_score(
        anomaly_patch_maps: Float[Tensor, "b c p p"],
        pred_patch_masks: Bool[Tensor, "b c p p"],
        outputs: Dict[str, Union[List, Tensor]],
        trainer,
):
    """Determines the label and anomaly score and adds them to `outputs`."""
    label_mapping = trainer.datamodule.label_mapping  # add for better visualization
    outputs["label_mapping"] = label_mapping

    # array of length `num_classes` with score for index of label and all other values 0
    pred_scores: Float[Tensor, "b c"] = torch.zeros(*anomaly_patch_maps.shape[:2], device=anomaly_patch_maps.device)
    pred_labels: List[int] = []

    batch_size, num_classes, _, _ = anomaly_patch_maps.shape
    for i in range(batch_size):
        bincount = torch.stack([pred_patch_masks[i, j].sum() for j in range(num_classes)])
        assert len(bincount) == num_classes == pred_patch_masks.shape[1]

        if num_classes == 1:  # binary classification
            label = 1 if bincount[0] > 0 else 0
            pred_scores[i][0] = anomaly_patch_maps[i, 0].max().item()
        else:  # multiclass
            if bincount[1:].any():  # any anomalous patches
                label = bincount[1:].argmax().item() + 1
                pred_scores[i][label] = anomaly_patch_maps[i, label].max().item()
            else:  # classify as good
                label = 0
                # use `.min()` for good map because the worst patch is most important
                pred_scores[i][label] = anomaly_patch_maps[i, label].min().item()
        pred_labels.append(label)

    outputs["pred_scores"] = pred_scores.squeeze()
    outputs["pred_labels"] = pred_labels
