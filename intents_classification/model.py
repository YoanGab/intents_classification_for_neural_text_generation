from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sadice import SelfAdjDiceLoss
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel

from intents_classification.datasets import SiliconeDataset


class Model:
    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: str,
        language_model_name: str,
        model_type: nn.Module,
        model_name: str,
        model_dir: Path,
        language_model_class: any = AutoModel,
        batch_size=8,
        loss: str = "cross_entropy",
        load: bool = True,
    ):
        self._load_datasets(dataset_name, tokenizer_name)
        self._load_dataloader(batch_size)

        self.language_model_name = language_model_name
        self.language_model = language_model_class.from_pretrained(language_model_name)
        self.embedding_dim = self.language_model.config.hidden_size

        self.model_name = model_name
        self.model_path = model_dir / f"{self.model_name}.pth"
        self.model_type = model_type
        self.loss = loss
        self.load = load
        self._reset_model()
        self._set_loss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _load_datasets(self, dataset_name: str, tokenizer_name: str):
        self.train_dataset = SiliconeDataset(
            set_type="train", dataset_name=dataset_name, tokenizer_name=tokenizer_name
        )
        self.eval_dataset = SiliconeDataset(
            set_type="validation",
            dataset_name=dataset_name,
            tokenizer_name=tokenizer_name,
        )
        self.test_dataset = SiliconeDataset(
            set_type="test", dataset_name=dataset_name, tokenizer_name=tokenizer_name
        )
        self.unique_labels = self.train_dataset.labels

    def _load_dataloader(self, batch_size: int):
        self.train_dataloader = DataLoader(
            self.train_dataset, shuffle=True, batch_size=batch_size
        )
        self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=batch_size)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size)

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
        self.model = self.model_type(
            self.train_dataset.tokenizer, len(self.unique_labels), self.embedding_dim
        )
        if self.load:
            self._load_model()

    def _load_model(self):
        if self.model_path.exists():
            print(f"Loading model {self.model_name}")
            self.model.load_state_dict(torch.load(self.model_path))
        else:
            print(f"Model {self.model_name} not found")

    def _save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def _get_class_weights(self):
        labels = []
        for batch in tqdm(self.train_dataloader):
            labels.extend(batch["labels"].numpy())
        self.class_weights = class_weight.compute_class_weight(
            "balanced", classes=range(len(self.unique_labels)), y=labels
        )

    def _plot_acc_loss(self):
        figure_path = Path("figs") / self.model_name
        figure_path.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots()
        ax.plot(range(len(self.total_loss)), self.total_loss)
        ax.set_title(f"Evolution of loss over {len(self.total_loss)} epochs")
        fig.savefig(figure_path / f"loss.png", format="png")
        fig, ax = plt.subplots()
        ax.plot(range(len(self.total_accuracy)), self.total_accuracy)
        ax.set_title(f"Evolution of accuracy over {len(self.total_accuracy)} epochs")
        fig.savefig(figure_path / f"acc.png", format="png")

    def train(
        self,
        nb_epoch: int,
        learning_rate: float = 5e-5,
        clip_grad_norm: float = 1.0,
        save_freq: int = 1000,
        warm_start: bool = True,
    ):
        if not warm_start:
            self._reset_model()

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        self.total_loss = []
        self.total_accuracy = []
        nb_val_correct = 0
        nb_val_total = 0
        progress_bar = tqdm(range(nb_epoch * len(self.train_dataloader)))
        self.model.train()
        for epoch in range(nb_epoch):
            for step, batch in enumerate(self.train_dataloader):
                b_input_ids = batch["input_ids"].to(self.device)
                b_input_mask = batch["attention_mask"].to(self.device)
                b_labels = batch["labels"].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(b_input_ids, attention_mask=b_input_mask)
                loss = self.criterion(outputs, b_labels)
                self.total_loss.append(loss.item())

                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)

                optimizer.step()

                preds = torch.argmax(outputs, dim=-1)
                preds = preds.detach().cpu().numpy()
                nb_val_correct += (
                    (preds == b_labels.detach().cpu().numpy()).sum().item()
                )
                nb_val_total += len(b_labels.detach().cpu().numpy())
                self.total_accuracy.append(nb_val_correct / nb_val_total)

                progress_bar.set_description(
                    f"loss: {round(self.total_loss[-1],3): .4f} | Avg loss: {np.mean(self.total_loss): .4f} | Avg Acc{100*self.total_accuracy[-1] : .2f}"
                )
                progress_bar.update(1)
                if step % save_freq == 0:
                    self._save_model()

            val_mean_loss, val_mean_accuracy = self.evaluate()
            print(
                f"Epoch {epoch+1}/{nb_epoch}, Train Loss: {self.total_loss[-1]:.4f}, Train Acc: {100*self.total_accuracy[-1] : .2f}, Val Loss: {val_mean_loss:.4f}, Val Acc: {val_mean_accuracy:.4f}"
            )
        self._save_model()
        self._plot_acc_loss()

    def evaluate(self):
        total_loss = []
        total_accuracy = []
        nb_val_correct = 0
        nb_val_total = 0
        progress_bar = tqdm(range(len(self.eval_dataloader)))
        self.model.eval()

        with torch.no_grad():
            for batch in self.eval_dataloader:
                b_input_ids = batch["input_ids"].to(self.device)
                b_input_mask = batch["attention_mask"].to(self.device)
                b_labels = batch["labels"].to(self.device)

                outputs = self.model(b_input_ids, attention_mask=b_input_mask)
                loss = self.criterion(outputs, b_labels)
                total_loss.append(loss.item())

                preds = torch.argmax(outputs, dim=-1)
                preds = preds.detach().cpu().numpy()
                nb_val_correct += (
                    (preds == b_labels.detach().cpu().numpy()).sum().item()
                )
                nb_val_total += len(b_labels.detach().cpu().numpy())
                total_accuracy.append(nb_val_correct / nb_val_total)

                progress_bar.update(1)

        val_mean_loss = np.mean(total_loss)
        val_mean_accuracy = total_accuracy[-1]

        self.model.train()
        return val_mean_loss, val_mean_accuracy

    def test(self):
        preds, trues = [], []
        for i, batch in tqdm(
            enumerate(self.test_dataloader),
            desc="testing",
            total=self.test_dataloader.__len__(),
        ):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            b_input_ids = batch["input_ids"].to(self.device)
            b_input_mask = batch["attention_mask"].to(self.device)
            b_labels = batch["labels"].to(self.device)

            with torch.no_grad():
                outputs = self.model(b_input_ids, attention_mask=b_input_mask)
            predictions = torch.argmax(outputs, dim=-1)
            preds.extend(predictions.cpu().detach().tolist())
            trues.extend(b_labels.cpu().detach().tolist())

        print(
            classification_report(
                np.array(trues).flatten(),
                np.array(preds).flatten(),
                target_names=len(self.unique_labels),
            )
        )

        cm = confusion_matrix(np.array(trues).flatten(), np.array(preds).flatten())
        df_cm = pd.DataFrame(
            cm, index=len(self.unique_labels), columns=len(self.unique_labels)
        )
        print(df_cm)
