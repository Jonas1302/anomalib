from typing import Any, Optional, Dict, Tuple, List

from jaxtyping import Bool, Float
import timm
import torch
import torchmetrics
from torch import Tensor

from anomalib.models import AnomalyModule
from anomalib.models.patchcore.torch_model import PatchcoreModel

from anomalib.models.patchcore.utils import process_pred_masks, process_label_and_score


def get_classifier(embedding_size: int, num_classes: int, hidden_size: int = 0, dropout: float = 0, flatten: bool = False):
    model = torch.nn.Sequential()

    if flatten:
        model.append(torch.nn.AdaptiveAvgPool2d((1, 1)))
        model.append(torch.nn.Flatten())

    if dropout:
        model.append(torch.nn.Dropout(dropout))

    if hidden_size == 0:
        hidden_size = embedding_size
    else:
        model.append(torch.nn.Linear(embedding_size, hidden_size))
        model.append(torch.nn.LeakyReLU())
        if dropout:
            model.append(torch.nn.Dropout(dropout))

    model.append(torch.nn.Linear(hidden_size, 1 if num_classes == 2 else num_classes))

    return model


class Classifier(AnomalyModule):
    def __init__(self, lr: float, dropout: float, num_classes: int, **kwargs):
        super().__init__()
        self.lr = lr
        self.dropout = dropout
        self.num_classes = num_classes
        self.loss = torch.nn.functional.cross_entropy if num_classes >= 2 else torch.nn.functional.binary_cross_entropy_with_logits
        self.accuracy = torchmetrics.Accuracy()

    def training_step(self, batch, batch_idx, log=True, log_prefix="") -> Dict:
        batch_size = len(batch["label"])
        predictions: Float[Tensor, "b c"] = self(batch["image"])
        # use index 0 for weight because there will be `batch_size` number of identical tensors concatenated together
        loss = self._calculate_loss(predictions, batch["label"], self.trainer.datamodule.train_data.images_per_class)

        if log:
            self.log(f"{log_prefix}loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        batch["loss"] = loss
        batch["pred_scores"] = predictions.detach().clone()  # the original tensor must remain unchanged for gradient computation
        if self.num_classes > 1:  # multiclass
            batch["pred_scores"] = torch.nn.functional.softmax(batch["pred_scores"], dim=-1)
        else:
            batch["pred_scores"] = torch.sigmoid(batch["pred_scores"]).squeeze()

        label_mapping = self.trainer.datamodule.label_mapping
        batch["label_mapping"] = label_mapping
        
        batch["pred_labels"] = []
        for i in range(batch_size):
            label = predictions[i].argmax()
            batch["pred_labels"].append(label_mapping[label.cpu().item()])

        return batch

    def _calculate_loss(self, predictions, targets, images_per_class):
        if self.num_classes == 1:
            # pos_weight is num_negative/num_positive
            # see https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
            loss = self.loss(predictions.flatten(), targets.float(), pos_weight=images_per_class[0] / images_per_class[1])
        else:
            loss = self.loss(predictions, targets, weight=1 / images_per_class)
        return loss

    def validation_step(self, batch, batch_idx) -> Dict:
        with torch.no_grad():
            return self.training_step(batch, batch_idx, log_prefix="val_")

    def predict_step(self, batch: Any, batch_idx: int, _dataloader_idx: Optional[int] = None) -> Dict:
        with torch.no_grad():
            return self.training_step(batch, batch_idx, log=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class TransferLearningClassifier(Classifier):
    def __init__(self, backbone: str, num_classes: int, freeze_batch_norm: bool, hidden_size: int, layers: List[str],
                 input_size: Tuple[int, int], use_mlp: bool = False, use_global_embedding: bool = False,
                 locally_aware_patch_features: bool = True, **kwargs):
        super().__init__(num_classes, **kwargs)
        self.freeze_batch_norm = freeze_batch_norm
        flatten = False

        if use_global_embedding:
            if backbone == "wide_resnet50_2":
                feature_size = 1536
            elif backbone.startswith("vit_base"):
                feature_size = 768 * len(layers)
            else:
                raise ValueError(f"unsupported backbone model {backbone}")

            self.patchcore_model = PatchcoreModel(
                input_size=input_size,
                layers=layers,
                backbone=backbone,
                locally_aware_patch_features=locally_aware_patch_features,
            )
            self.pretrained_model = self.patchcore_model.get_embedding
            self.freeze_batch_norm = False  # is always frozen by `PatchcoreModel.feature_extractor`
            flatten = True
        else:
            if backbone == "wide_resnet50_2":
                self.pretrained_model: torch.nn.Module = timm.create_model(backbone, pretrained=True)
                self.pretrained_model.fc = torch.nn.Identity()  # replace last fully connected layer
                self.pretrained_model.requires_grad_(False)
                feature_size = 2048
            elif backbone.startswith("vit_base"):
                self.pretrained_model: torch.nn.Module = timm.create_model(backbone, pretrained=True)
                self.pretrained_model.head = torch.nn.Identity()  # replace last fully connected layer
                self.pretrained_model.requires_grad_(False)
                feature_size = 768
            else:
                raise ValueError(f"unsupported backbone model '{backbone}'")

        self.classifier = get_classifier(feature_size, num_classes, hidden_size if use_mlp else 0, self.dropout, flatten)

    def forward(self, input_tensor: Tensor) -> Tensor:
        if self.freeze_batch_norm:
            self.pretrained_model.eval()  # gets set to 'train' whenever 'self' is set to 'train', so this must be called every time
        with torch.no_grad():
            output = self.pretrained_model(input_tensor)
        return self.classifier(output)


class PatchBasedClassifier(Classifier):
    def __init__(
            self,
            backbone: str,
            num_classes: int,
            input_size: Tuple[int, int],
            hidden_size: int,
            layers: List[str],
            use_threshold: bool,
            **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.use_threshold = use_threshold
        assert self.use_threshold or num_classes != 1, f"{self.use_threshold=}, {num_classes=}"

        if backbone == "wide_resnet50_2":
            self.embedding_size = 1536
        elif backbone.startswith("vit_base"):
            self.embedding_size = 768 * len(layers)
        else:
            raise ValueError(f"unsupported backbone model {backbone}")

        self.backbone = PatchcoreModel(
            input_size=input_size,
            layers=layers,
            backbone=backbone,
        )

        self.classifier = get_classifier(self.embedding_size, self.num_classes, hidden_size, self.dropout)

    def forward(self, input_tensor: Float[Tensor, "b _ w h"]) -> Float[Tensor, "b c"]:
        return self.classifier(input_tensor)

    def validation_step(self, batch, batch_idx) -> Dict:
        with torch.no_grad():
            if len(batch["image"].shape) == 4:  # revalidation with images
                return self.predict_step(batch, batch_idx)
            return self.training_step(batch, batch_idx, log=True, log_prefix="val_")

    def predict_step(self, batch: Any, batch_idx: int, _dataloader_idx: Optional[int] = None, log=False) -> Dict:
        anomaly_maps, anomaly_patch_maps = self.extract_anomaly_maps(batch)
        threshold = self.image_threshold if self.use_threshold else None
        pred_masks, pred_patch_masks = process_pred_masks(anomaly_maps, anomaly_patch_maps, batch, threshold)
        process_label_and_score(anomaly_patch_maps, pred_patch_masks, batch, self.trainer)

        loss = self._calculate_loss(batch["pred_scores"], batch["label"], batch["images_per_class"][0])
        batch["loss"] = loss

        if log:
            self.log(f"val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch["label"]))

        return batch

    def extract_anomaly_maps(self, batch):
        anomaly_maps: List[Float[Tensor, "1 c w h"]] = []
        anomaly_patch_maps: List[Float[Tensor, "1 c p p"]] = []

        for i in range(len(batch["image"])):
            embedding: Float[Tensor, "1 f p p"] = self.extract_embeddings(batch["image"][i])
            _batch_size, num_features, width, height = embedding.shape
            assert _batch_size == 1 and num_features == self.embedding_size, f"{_batch_size=}; {num_features=}"
            embedding: Float[Tensor, "p*p f"] = PatchcoreModel.reshape_embedding(embedding)

            with torch.no_grad():
                prediction: Float[Tensor, "p*p c"] = self(embedding)
                if self.num_classes > 1:  # apply softmax, but only for non-binary classification
                    prediction = torch.nn.functional.softmax(prediction, dim=-1)
                else:
                    prediction = torch.sigmoid(prediction)

            prediction: Float[Tensor, "1 c p p"] = prediction.reshape(1, width, height, self.num_classes).permute(0, 3, 1, 2)
            anomaly_patch_maps.append(prediction)  # map of class probabilities
            anomaly_maps.append(self.backbone.anomaly_map_generator(prediction.permute(1, 0, 2, 3)).permute(1, 0, 2, 3))

        anomaly_maps: Float[Tensor, "b c w h"] = torch.concat(anomaly_maps)
        anomaly_patch_maps: Float[Tensor, "b c p p"] = torch.concat(anomaly_patch_maps)

        batch["anomaly_maps"] = anomaly_maps

        return anomaly_maps, anomaly_patch_maps

    def extract_embeddings(self, input_tensor: Float[Tensor, "b c w h"]) -> Float[Tensor, "b f p p"]:
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
        return self.backbone.get_embedding(input_tensor)

    def overwrite_backbone(self, backbone):
        self.backbone.overwrite_backbone(backbone)
