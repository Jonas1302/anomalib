from typing import Any, Optional, Dict, Tuple

import timm
import torch
import torchmetrics
from torch import Tensor

from anomalib.models import AnomalyModule
from anomalib.models.patchcore.torch_model import PatchcoreModel


class Classifier(AnomalyModule):
    def __init__(self, lr: float, **kwargs):
        super().__init__()
        self.lr = lr
        self.loss = torch.nn.functional.cross_entropy
        self.accuracy = torchmetrics.Accuracy()
        self.image_threshold = None
        self.pixel_threshold = None
    
    def _compute_adaptive_threshold(self, outputs):
        pass
    
    def training_step(self, batch, batch_idx, log=True, log_prefix="") -> Dict:
        batch_size = len(batch["label"])
        predictions: Tensor = self(batch["image"])
        # use index 0 for weight because there will be `batch_size` number of identical tensors concatenated together
        loss = self.loss(predictions, batch["label"], weight=1 / batch["images_per_class"][0])
        accuracy = self.accuracy(predictions, batch["label"])

        if log:
            self.log(f"{log_prefix}loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
            self.log(f"{log_prefix}accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        
        batch["loss"] = loss
        batch["pred_scores"] = predictions.detach().clone()  # the original tensor must remain unchanged for gradient computation
        
        label_mapping = self.trainer.datamodule.label_mapping
        batch["label_mapping"] = label_mapping
        
        batch["pred_labels"] = []
        # pred_scores_normed = (prediction - self.image_threshold.value) / (1 - self.image_threshold.value)
        
        for i in range(batch_size):
            # label = 0 if pred_scores_normed[i].max() < 0 else pred_scores_normed[i].argmax().item()
            label = predictions[i].argmax()
            batch["pred_labels"].append(label_mapping[label.cpu().item()])
            # batch["pred_scores"][i, torch.arange(0, len(batch["pred_scores"][i])) != label] = 0  # set all to zero except label
        
        return batch
    
    def validation_step(self, batch, batch_idx) -> Dict:
        with torch.no_grad():
            return self.training_step(batch, batch_idx, log_prefix="val_")

    def predict_step(self, batch: Any, batch_idx: int, _dataloader_idx: Optional[int] = None) -> Dict:
        with torch.no_grad():
            return self.training_step(batch, batch_idx, log=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

class ResnetClassifier(Classifier):
    def __init__(self, backbone: str, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        
        if backbone == "wide_resnet50_2":
            self.pretrained_model: torch.nn.Module = timm.create_model(backbone, pretrained=True)
            self.pretrained_model.fc = torch.nn.Identity()  # replace last fully connected layer
            self.pretrained_model.requires_grad_(False)
            self.pretrained_model.eval()  # TODO good or bad idea? Does it actually make a difference?
            self.classifier_model = torch.nn.Sequential(
                torch.nn.Linear(2048, num_classes),
            )
        else:
            raise ValueError(f"unsupported backbone model '{backbone}'")
    
    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.pretrained_model(input_tensor)
        return self.classifier_model(output)


class PatchBasedClassifier(Classifier):
    def __init__(
            self,
            backbone: str,
            num_classes: int,
            input_size: Tuple[int, int],
            hidden_size: int,
            **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes

        if backbone == "wide_resnet50_2":
            self.backbone = PatchcoreModel(
                input_size=input_size,
                layers=["layer2", "layer3"],
                backbone=backbone,
            )
            self.embedding_size = 1536
        elif backbone.startswith("vit_base"):
            self.backbone = PatchcoreModel(
                input_size=input_size,
                layers=[5, 9],
                backbone=backbone,
            )
            self.embedding_size = 768 * 2
        else:
            raise ValueError(f"unsupported backbone model {backbone}")

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, num_classes),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self.classifier(input_tensor)

    def validation_step(self, batch, batch_idx) -> Dict:
        with torch.no_grad():
            return self.training_step(batch, batch_idx, log=True, log_prefix="val_")

    def predict_step(self, batch: Any, batch_idx: int, _dataloader_idx: Optional[int] = None, log=False) -> Dict:
        predictions = []
        anomaly_maps = []

        for i in range(len(batch["image"])):
            embedding = self.extract_embeddings(batch["image"][i])
            batch_size, num_features, width, height = embedding.shape
            assert batch_size == 1 and num_features == self.embedding_size, f"{batch_size=}; {num_features=}"
            embedding = PatchcoreModel.reshape_embedding(embedding)  # shape: [1 * width * height, num_features]
            prediction: Tensor = self(embedding).reshape(1, width, height, self.num_classes)
            predictions.append(prediction)  # map of class probabilities
            anomaly_maps.append(self.backbone.anomaly_map_generator(prediction.permute(3, 0, 1, 2)).permute(1, 0, 2, 3))

        predictions = torch.concat(predictions)
        batch["anomaly_maps"] = torch.concat(anomaly_maps)
        batch["pred_masks"] = batch["anomaly_maps"] > 0.5

        label_mapping = self.trainer.datamodule.label_mapping  # add for better visualization
        batch["label_mapping"] = label_mapping

        batch["pred_scores"] = []
        batch["pred_labels"] = []
        for i in range(len(predictions)):
            bincount = predictions[i].argmax(dim=-1).flatten().bincount()
            if bincount[1:].any():  # any anomalous patches
                label = bincount[1:].argmax().item() + 1
            else:
                label = 0
            # array of length `num_classes` with max value for index of label and all other values 0
            batch["pred_scores"].append([predictions[i, :, :, j].max().item() if j == label else 0 for j in range(len(label_mapping))])
            batch["pred_labels"].append(label_mapping[label])
        batch["pred_scores"] = torch.tensor(batch["pred_scores"], device=self.device)

        loss = self.loss(batch["pred_scores"], batch["label"], weight=1 / batch["images_per_class"][0])
        accuracy = self.accuracy(batch["pred_scores"], batch["label"])
        batch["loss"] = loss

        if log:
            self.log(f"val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch["label"]))
            self.log(f"val_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch["label"]))

        return batch

    def extract_embeddings(self, input_tensor: Tensor) -> Tensor:
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
        return self.backbone.get_embedding(input_tensor)
