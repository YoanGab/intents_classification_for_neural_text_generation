import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from intents_classification.loaders import DatasetLoader, ModelLoader
from .utils import DataType, FigureType, flat_accuracy, plot_comparison


class Trainer:
    def __init__(self, datasetLoader: DatasetLoader, modelLoader: ModelLoader):
        self.datasetLoader = datasetLoader
        self.modelLoader = modelLoader

    def train(
        self,
        nb_epoch: int,
        learning_rate: float = 5e-5,
        clip_grad_norm: float = 1.0,
        save_freq: int = 1000,
        save: bool = False,
        warm_start: bool = True,
        scheduler: bool = False,
        checkpoints: bool = False,
    ):
        total_steps = len(self.datasetLoader.train_dataloader) * nb_epoch

        if not warm_start:
            self.modelLoader._reset_model()

        optimizer = AdamW(self.modelLoader.model.parameters(), lr=learning_rate)

        if scheduler:
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=total_steps
            )

        self.train_loss_per_batch = []
        self.train_accuracy_per_batch = []
        self.val_loss_per_epoch = []
        self.val_accuracy_per_epoch = []
        self.train_accuracy_per_epoch = []
        self.train_loss_per_epoch = []
        progress_bar = tqdm(range(total_steps))

        self.modelLoader.model.train()
        for epoch in range(nb_epoch):
            train_loss_over_one_epoch = []
            train_accuracy_over_one_epoch = []
            for step, batch in enumerate(self.datasetLoader.train_dataloader):
                b_input_ids = batch["input_ids"].to(self.modelLoader.device)
                b_input_mask = batch["attention_mask"].to(self.modelLoader.device)
                b_labels = batch["labels"].to(self.modelLoader.device)

                optimizer.zero_grad()

                outputs = self.modelLoader.model(
                    input_ids=b_input_ids,
                    attention_mask=b_input_mask,
                    token_type_ids=None,
                    labels=b_labels,
                )

                if self.modelLoader.model_type == "auto":
                    loss = outputs.loss
                    outputs = outputs.logits
                else:
                    loss = self.modelLoader.criterion(outputs, b_labels)

                self.train_loss_per_batch.append(loss.item())
                train_loss_over_one_epoch.append(loss.item())

                loss.backward()

                if clip_grad_norm != None:
                    nn.utils.clip_grad_norm_(
                        self.modelLoader.model.parameters(), clip_grad_norm
                    )

                detached_outputs = outputs.detach().cpu().numpy()
                detached_labels = b_labels.detach().cpu().numpy()
                b_accuracy = flat_accuracy(detached_outputs, detached_labels)

                self.train_accuracy_per_batch.append(b_accuracy)
                train_accuracy_over_one_epoch.append(b_accuracy)

                optimizer.step()
                if scheduler:
                    self.scheduler.step()

                progress_bar.set_description(
                    f"loss: {self.train_loss_per_batch[-1]: 4f} | Acc: {100*self.train_accuracy_per_batch[-1]: 3f} | Avg loss: {np.mean(self.train_loss_per_batch): .4f} | Avg Acc{100*np.mean(self.train_accuracy_per_batch) : .3f}"
                )
                progress_bar.update(1)
                if (step % save_freq == 0) and (step != 0) and checkpoints:
                    self.modelLoader._save_model(checkpoint=True, epoch=epoch)

            avg_current_train_loss = np.mean(train_loss_over_one_epoch)
            self.train_loss_per_epoch.append(avg_current_train_loss)

            avg_current_train_accuracy = np.mean(train_accuracy_over_one_epoch)
            self.train_accuracy_per_epoch.append(avg_current_train_accuracy)

            avg_val_loss, avg_val_accuracy = self.evaluate()
            self.val_loss_per_epoch.append(avg_val_loss)
            self.val_accuracy_per_epoch.append(avg_val_accuracy)

            print(
                f"Epoch {epoch+1}/{nb_epoch}, Train Loss: {avg_current_train_loss:.4f}, Train Acc: {100*avg_current_train_accuracy: .3f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {100*avg_val_accuracy:.3f}"
            )

        if save:
            self.modelLoader._save_model()
        if nb_epoch > 1:
            plot_comparison(
                self.modelLoader.model_name,
                [self.train_loss_per_epoch, self.val_loss_per_epoch],
                FigureType.TRAIN_VAL,
                DataType.LOSS,
                True,
            )

            plot_comparison(
                self.modelLoader.model_name,
                [self.train_accuracy_per_epoch, self.val_accuracy_per_epoch],
                FigureType.TRAIN_VAL,
                DataType.ACCURACY,
                True,
            )

        plot_comparison(
            self.modelLoader.model_name,
            [self.train_accuracy_per_batch],
            FigureType.TRAIN,
            DataType.ACCURACY,
            False,
        )

    def evaluate(self):
        loss_per_batch = []
        accuracy_per_batch = []
        progress_bar = tqdm(range(len(self.datasetLoader.eval_dataloader)))
        self.modelLoader.model.eval()

        with torch.no_grad():
            for batch in self.datasetLoader.eval_dataloader:
                b_input_ids = batch["input_ids"].to(self.modelLoader.device)
                b_input_mask = batch["attention_mask"].to(self.modelLoader.device)
                b_labels = batch["labels"].to(self.modelLoader.device)

                outputs = self.modelLoader.model(
                    input_ids=b_input_ids,
                    attention_mask=b_input_mask,
                    token_type_ids=None,
                    labels=b_labels,
                )

                if self.modelLoader.model_type == "auto":
                    loss = outputs.loss
                    outputs = outputs.logits
                else:
                    loss = self.modelLoader.criterion(outputs, b_labels)
                loss_per_batch.append(loss.item())

                detached_outputs = outputs.detach().cpu().numpy()
                detached_labels = b_labels.detach().cpu().numpy()

                b_accuracy = flat_accuracy(detached_outputs, detached_labels)
                accuracy_per_batch.append(b_accuracy)

                progress_bar.update(1)

        self.modelLoader.model.train()
        return np.mean(loss_per_batch), np.mean(accuracy_per_batch)

    def test(self):
        preds, trues = [], []
        for i, batch in tqdm(
            enumerate(self.datasetLoader.test_dataloader),
            desc="testing",
            total=self.datasetLoader.test_dataloader.__len__(),
        ):
            batch = {k: v.to(self.modelLoader.device) for k, v in batch.items()}
            b_input_ids = batch["input_ids"].to(self.modelLoader.device)
            b_input_mask = batch["attention_mask"].to(self.modelLoader.device)
            b_labels = batch["labels"].to(self.modelLoader.device)

            with torch.no_grad():
                outputs = self.modelLoader.model(
                    input_ids=b_input_ids,
                    attention_mask=b_input_mask,
                    token_type_ids=None,
                    labels=b_labels,
                )
                if self.modelLoader.model_type == "auto":
                    outputs = outputs.logits
                predictions = np.argmax(outputs.detach().cpu().numpy(), axis=1)

            preds.extend(predictions)
            trues.extend(b_labels.cpu().detach().tolist())

        report_df = pd.DataFrame(
            classification_report(
                np.array(trues).flatten(),
                np.array(preds).flatten(),
                target_names=self.datasetLoader.unique_labels,
                output_dict=True,
            )
        ).transpose()

        report_df.to_csv(self.modelLoader.model_dir / "classification_report.csv")

        cm = confusion_matrix(np.array(trues).flatten(), np.array(preds).flatten())
        df_cm = pd.DataFrame(
            cm,
            index=self.datasetLoader.unique_labels,
            columns=self.datasetLoader.unique_labels,
        )
        df_cm.to_csv(self.modelLoader.model_dir / "confusion_matrix.csv")

        print("Classification Report")
        print(report_df)
        print("Confusion Matrix")
        print(df_cm)
