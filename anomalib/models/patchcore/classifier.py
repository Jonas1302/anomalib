from typing import Any, Optional, Dict, Tuple, List

from jaxtyping import Bool, Float
import timm
import torch
import torchmetrics
from torch import Tensor

from anomalib.models import AnomalyModule
from anomalib.models.patchcore.torch_model import PatchcoreModel

from anomalib.models.patchcore.utils import process_pred_masks, process_label_and_score


class Classifier(AnomalyModule):
    def __init__(self, lr: float, **kwargs):
        super().__init__()
        self.lr = lr
        self.loss = torch.nn.functional.cross_entropy
        self.accuracy = torchmetrics.Accuracy()

    def training_step(self, batch, batch_idx, log=True, log_prefix="") -> Dict:
        batch_size = len(batch["label"])
        predictions: Float[Tensor, "b c"] = self(batch["image"])
        # use index 0 for weight because there will be `batch_size` number of identical tensors concatenated together
        loss = self.loss(predictions, batch["label"], weight=1 / batch["images_per_class"][0])
        accuracy = self.accuracy(predictions, batch["label"])

        if log:
            self.log(f"{log_prefix}loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
            self.log(f"{log_prefix}accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        
        batch["loss"] = loss
        batch["pred_scores"] = torch.nn.functional.softmax(predictions.detach().clone(), dim=-1)  # the original tensor must remain unchanged for gradient computation
        
        label_mapping = self.trainer.datamodule.label_mapping
        batch["label_mapping"] = label_mapping
        
        batch["pred_labels"] = []
        for i in range(batch_size):
            label = predictions[i].argmax()
            batch["pred_labels"].append(label_mapping[label.cpu().item()])

        return batch
    
    def validation_step(self, batch, batch_idx) -> Dict:
        with torch.no_grad():
            return self.training_step(batch, batch_idx, log_prefix="val_")

    def predict_step(self, batch: Any, batch_idx: int, _dataloader_idx: Optional[int] = None) -> Dict:
        with torch.no_grad():
            return self.training_step(batch, batch_idx, log=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

class TransferLearningClassifier(Classifier):
    def __init__(self, backbone: str, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        
        if backbone == "wide_resnet50_2":
            self.pretrained_model: torch.nn.Module = timm.create_model(backbone, pretrained=True)
            self.pretrained_model.fc = torch.nn.Identity()  # replace last fully connected layer
            self.pretrained_model.requires_grad_(False)
            self.classifier_model = torch.nn.Sequential(
                torch.nn.Linear(2048, num_classes),
            )
        else:
            raise ValueError(f"unsupported backbone model '{backbone}'")
    
    def forward(self, input_tensor: Tensor) -> Tensor:
        if self.freeze_batch_norm:
            self.pretrained_model.eval()  # gets set to 'train' whenever 'self' is set to 'train', so this must be called every time
        output = self.pretrained_model(input_tensor)
        return self.classifier_model(output)


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
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.use_threshold = use_threshold

        if backbone == "wide_resnet50_2":
            self.backbone = PatchcoreModel(
                input_size=input_size,
                layers=layers,
                backbone=backbone,
            )
            self.embedding_size = 1536
        elif backbone.startswith("vit_base"):
            self.backbone = PatchcoreModel(
                input_size=input_size,
                layers=layers,
                backbone=backbone,
            )
            self.embedding_size = 768 * len(layers)
        else:
            raise ValueError(f"unsupported backbone model {backbone}")

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, num_classes),
        )

    def forward(self, input_tensor: Float[Tensor, "b _ w h"]) -> Float[Tensor, "b c"]:
        return self.classifier(input_tensor)

    def validation_step(self, batch, batch_idx) -> Dict:
        with torch.no_grad():
            return self.training_step(batch, batch_idx, log=True, log_prefix="val_")

    def predict_step(self, batch: Any, batch_idx: int, _dataloader_idx: Optional[int] = None, log=False) -> Dict:
        anomaly_maps, anomaly_patch_maps = self.extract_anomaly_maps(batch)
        threshold = self.image_threshold if self.use_threshold else None
        pred_masks, pred_patch_masks = process_pred_masks(anomaly_maps, anomaly_patch_maps, batch, threshold)
        process_label_and_score(anomaly_patch_maps, pred_patch_masks, batch, self.trainer)

        loss = self.loss(batch["pred_scores"], batch["label"], weight=1 / batch["images_per_class"][0])
        accuracy = self.accuracy(batch["pred_scores"], batch["label"])
        batch["loss"] = loss

        if log:
            self.log(f"val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch["label"]))
            self.log(f"val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch["label"]))

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
                prediction: Float[Tensor, "p*p c"] = torch.nn.functional.softmax(self(embedding), dim=-1)

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
