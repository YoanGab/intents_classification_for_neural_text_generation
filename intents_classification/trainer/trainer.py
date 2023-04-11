from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix

import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from .dataset_uploader import DatasetUploader
from .model_loader import ModelLoader
from .utils import flat_accuracy, plot_acc_loss

class Trainer:
    def __init__(self, datasetLoader: DatasetUploader, modelLoader: ModelLoader):
        self.datasetLoader = datasetLoader
        self.modelLoader = modelLoader
        
    def train(
        self,
        nb_epoch: int,
        learning_rate: float = 5e-5,
        clip_grad_norm: float = 1.0,
        save_freq: int = 1000,
        warm_start: bool = True,
        scheduler: bool = False,
    ):  
        total_steps = len(self.datasetLoader.train_dataloader) * nb_epoch
       
        if not warm_start:
            self.modelLoader._reset_model()

        optimizer = AdamW(self.modelLoader.model.parameters(), lr=learning_rate)

        if scheduler:
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps = 0,
                num_training_steps = total_steps
            )

        self.total_loss = []
        self.total_accuracy = []
        nb_val_correct = 0
        nb_val_total = 0
        progress_bar = tqdm(range(total_steps))

        self.modelLoader.model.train()
        for epoch in range(nb_epoch):
            for step, batch in enumerate(self.datasetLoader.train_dataloader):
                b_input_ids = batch["input_ids"].to(self.modelLoader.device)
                b_input_mask = batch["attention_mask"].to(self.modelLoader.device)
                b_labels = batch["labels"].to(self.modelLoader.device)

                optimizer.zero_grad()

                outputs = self.modelLoader.model(
                    input_ids = b_input_ids, 
                    attention_mask=b_input_mask,
                    token_type_ids=None, 
                    labels=b_labels
                )

                if self.modelLoader.model_type == "auto":
                    loss = outputs.loss
                    outputs = outputs.logits
                else:
                    loss = self.modelLoader.criterion(outputs, b_labels)

                self.total_loss.append(loss.item())

                loss.backward()

                nn.utils.clip_grad_norm_(self.modelLoader.model.parameters(), clip_grad_norm)

                optimizer.step()
                if scheduler:
                    self.scheduler.step()
                                    

                detached_outputs = outputs.detach().cpu().numpy()
                detached_labels = b_labels.detach().cpu().numpy()
                nb_val_correct += flat_accuracy(detached_outputs, detached_labels)
                nb_val_total += len(detached_labels)

                self.total_accuracy.append(nb_val_correct / nb_val_total)

                progress_bar.set_description(
                    f"loss: {round(self.total_loss[-1],3): .4f} | Avg loss: {np.mean(self.total_loss): .4f} | Avg Acc{100*self.total_accuracy[-1] : .2f}"
                )
                progress_bar.update(1)
                if (step % save_freq == 0) and (step != 0):
                    self.modelLoader._save_model(checkpoint=True, epoch=epoch)

            val_mean_loss, val_mean_accuracy = self.evaluate()
            print(
                f"Epoch {epoch+1}/{nb_epoch}, Train Loss: {self.total_loss[-1]:.4f}, Train Acc: {100*self.total_accuracy[-1] : .2f}, Val Loss: {val_mean_loss:.4f}, Val Acc: {val_mean_accuracy:.4f}"
            )
        self.modelLoader._save_model()
        plot_acc_loss(self.modelLoader.model_name, self.total_loss, self.total_accuracy)

    def evaluate(self):
        total_loss = []
        total_accuracy = []
        nb_val_correct = 0
        nb_val_total = 0
        progress_bar = tqdm(range(len(self.datasetLoader.eval_dataloader)))
        self.model.eval()

        with torch.no_grad():
            for batch in self.datasetLoader.eval_dataloader:
                b_input_ids = batch["input_ids"].to(self.modelLoader.device)
                b_input_mask = batch["attention_mask"].to(self.modelLoader.device)
                b_labels = batch["labels"].to(self.modelLoader.device)

                outputs = self.modelLoader.model(
                    input_ids = b_input_ids, 
                    attention_mask=b_input_mask,
                    token_type_ids=None, 
                    labels=b_labels
                )

                if self.modelLoader.model_type == "auto":
                    loss = outputs.loss
                    outputs = outputs.logits
                else:
                    loss = self.modelLoader.criterion(outputs, b_labels)
                total_loss.append(loss.item())


                detached_outputs = outputs.detach().cpu().numpy()
                detached_labels = b_labels.detach().cpu().numpy()
                nb_val_correct += flat_accuracy(detached_outputs, detached_labels)
                nb_val_total += len(detached_labels)

                total_accuracy.append(nb_val_correct / nb_val_total)

                progress_bar.update(1)

        val_mean_loss = np.mean(total_loss)
        val_mean_accuracy = total_accuracy[-1]

        self.modelLoader.model.train()
        return val_mean_loss, val_mean_accuracy

    def test(self):
        preds, trues = [], []
        for i, batch in tqdm(
            enumerate(self.datasetLoader.test_dataloader),
            desc="testing",
            total=self.datasetLoader.test_dataloader.__len__(),
        ):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            b_input_ids = batch["input_ids"].to(self.modelLoader.device)
            b_input_mask = batch["attention_mask"].to(self.modelLoader.device)
            b_labels = batch["labels"].to(self.modelLoader.device)

            with torch.no_grad():
                outputs = self.modelLoader.model(
                    input_ids = b_input_ids, 
                    attention_mask=b_input_mask,
                    token_type_ids=None, 
                    labels=b_labels
                )
                if self.modelLoader.model_type == "auto":
                    outputs = outputs.logits
                predictions = np.argmax(outputs.logits.detach().cpu().numpy(), axis=1)

            preds.extend(predictions)
            trues.extend(b_labels.cpu().detach().tolist())

        print(
            classification_report(
                np.array(trues).flatten(),
                np.array(preds).flatten(),
                target_names=self.unique_labels,
            )
        )

        cm = confusion_matrix(np.array(trues).flatten(), np.array(preds).flatten())
        df_cm = pd.DataFrame(
            cm, index=self.unique_labels, columns=self.unique_labels
        )
        print(df_cm)