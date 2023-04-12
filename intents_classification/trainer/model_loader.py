from pathlib import Path

import torch
import torch.nn as nn
from sadice import SelfAdjDiceLoss
from sklearn.utils import class_weight
from tqdm import tqdm
from transformers import AutoModel, BertForSequenceClassification

from .dataset_uploader import DatasetUploader


class ModelLoader:
    def __init__(
        self,
        datasetLoader: DatasetUploader,
        language_model_name: str,
        model_type: nn.Module or str,
        model_name: str,
        language_model_class: any = AutoModel,
        loss: str = "cross_entropy",
        load: bool = True,
    ):
        self.datasetLoader = datasetLoader
        self.language_model_name = language_model_name
        self.language_model = language_model_class.from_pretrained(language_model_name)
        self.embedding_dim = self.language_model.config.hidden_size

        self.model_name = model_name
        self.model_dir = (
            Path(f"intents_classification")
            / "models"
            / "model_checkpoints"
            / f"{self.model_name}"
        )
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints = [checkpoint for checkpoint in self.model_dir.glob("*.pth")]
        self.model_type = model_type
        self.loss = loss
        self.load = load
        self._reset_model()
        if self.model_type != "auto":
            self._set_loss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _set_loss(self):
        if self.loss == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        elif self.loss == "weighted_cross_entropy":
            self._get_class_weights()
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        elif self.loss == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.loss == "dice":
            self.criterion = SelfAdjDiceLoss()
        else:
            raise ValueError("Loss not supported")

    def _reset_model(self):
        if type(self.model_type) != str:
            self.model = self.model_type(
                self.language_model,
                len(self.datasetLoader.unique_labels),
                self.embedding_dim,
            )
        elif self.model_type == "auto":
            self.model = BertForSequenceClassification.from_pretrained(
                self.language_model_name,
                num_labels=len(self.datasetLoader.unique_labels),
                output_attentions=False,
                output_hidden_states=False,
            )
        else:
            raise ValueError("Model type not supported")
        if self.load:
            self._load_model()

    def _load_model(self):
        self.model_path = self.model_dir / f"{self.model_name}.pth"
        if self.model_path.exists():
            print(f"Loading model {self.model_name}")
            self.model.load_state_dict(torch.load(self.model_path))
        else:
            print(f"Model {self.model_name} not found")

    def _save_model(self, checkpoint: bool = False, epoch: int = 0):
        if checkpoint == False:
            self.model_path = self.model_dir / f"{self.model_name}.pth"
        torch.save(
            self.model.state_dict(),
            self.model_path
            if not checkpoint
            else self.model_dir / f"{self.model_name}_{epoch}.pth",
        )

    def _get_class_weights(self):
        labels = []
        for batch in tqdm(self.datasetLoader.train_dataloader):
            labels.extend(batch["labels"].numpy())
        self.class_weights = class_weight.compute_class_weight(
            "balanced", classes=range(len(self.datasetLoader.unique_labels)), y=labels
        )
