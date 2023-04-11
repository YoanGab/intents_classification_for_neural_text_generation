# -*- coding: utf-8 -*-
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             roc_curve)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer

from intents_classification.datasets import load_dataset, load_metric


class SiliconeDataset(Dataset):
    def __init__(self, name, name_dataset="dyda_da"):
        self.dataset = load_dataset("silicone", name_dataset)[name]
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", use_fast=True
        )
        self._tokenize_dataset()
        self.labels = self.dataset.features["Label"].names

    def _tokenize_dataset(self):
        self.tokenized_datasets = self.dataset.map(
            lambda example: self.tokenizer(
                example["Utterance"], padding="max_length", truncation=True
            ),
            batched=True,
        )
        self.tokenized_datasets = self.tokenized_datasets.remove_columns(
            ["Dialogue_ID", "Dialogue_Act", "Idx", "Utterance", "token_type_ids"]
        )
        self.tokenized_datasets = self.tokenized_datasets.rename_column(
            "Label", "labels"
        )
        self.tokenized_datasets.set_format("torch")

    def __getitem__(self, idx):
        return self.tokenized_datasets[idx]

    def __len__(self):
        return len(self.tokenized_datasets)


# Add a multi-layer neural network on top of BERT
class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_classes, embedding_dim):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output[1]  # Use the output of [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class BertClassifier2(nn.Module):
    def __init__(self, bert_model, num_classes, embedding_dim):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            # nn.Linear(embedding_dim, 512),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            # nn.Linear(512, num_classes),
            nn.Linear(embedding_dim, num_classes),
            nn.Softmax(dim=0),
        )

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output[1]  # Use the output of [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# Train the model
def train(model, nb_epoch, train_dataloader, eval_dataloader, device, names):
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    total_loss = []
    total_accuracy = []
    nb_val_correct = 0
    nb_val_total = 0
    progress_bar = tqdm(range(len(train_dataloader)))
    model.train()
    for epoch in range(nb_epoch):
        for step, batch in enumerate(train_dataloader):
            # Load batch to GPU
            b_input_ids = batch["input_ids"].to(device)
            b_input_mask = batch["attention_mask"].to(device)
            b_labels = batch["labels"].to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            loss = criterion(outputs, b_labels)
            total_loss.append(loss.item())

            # Backward pass
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update tracking variables
            preds = torch.argmax(outputs, dim=-1)
            preds = preds.detach().cpu().numpy()
            nb_val_correct += (preds == b_labels.detach().cpu().numpy()).sum().item()
            nb_val_total += len(b_labels.detach().cpu().numpy())
            total_accuracy.append(nb_val_correct / nb_val_total)

            progress_bar.set_description(
                f"loss: {round(total_loss[-1],3): .4f} | Avg loss: {np.mean(total_loss): .4f} | Avg Acc{100*total_accuracy[-1] : .2f}"
            )
            progress_bar.update(1)
            if step % 100 == 0:
                torch.save(model.state_dict(), "models/bert_classifier_weights.pth")
        val_loss, val_accuracy = evaluate(model, eval_dataloader, device, names)
        print(
            f"Epoch {epoch+1}/{nb_epoch}, Train Loss: {total_loss[-1]:.4f}, Train Acc: {100*total_accuracy[-1] : .2f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
        )
        torch.save(model.state_dict(), "models/bert_classifier_weights.pth")
        return total_loss, total_accuracy


def evaluate(model, eval_dataloader, device, names):
    total_loss = []
    total_accuracy = []
    nb_val_correct = 0
    nb_val_total = 0
    progress_bar = tqdm(range(len(eval_dataloader)))
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in eval_dataloader:
            # Load batch to GPU
            b_input_ids = batch["input_ids"].to(device)
            b_input_mask = batch["attention_mask"].to(device)
            b_labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            loss = criterion(outputs, b_labels)
            total_loss.append(loss.item())

            # Update tracking variables
            preds = torch.argmax(outputs, dim=-1)
            preds = preds.detach().cpu().numpy()
            nb_val_correct += (preds == b_labels.detach().cpu().numpy()).sum().item()
            nb_val_total += len(b_labels.detach().cpu().numpy())
            total_accuracy.append(nb_val_correct / nb_val_total)

            progress_bar.update(1)

    val_mean_loss = np.mean(total_loss)
    val_mean_accuracy = total_accuracy[-1]
    return val_mean_loss, val_mean_accuracy


def test(model, test_dataloader, device, names):
    preds, trues = [], []
    for i, batch in tqdm(
        enumerate(test_dataloader), desc="evaluating", total=test_dataloader.__len__()
    ):
        batch = {k: v.to(device) for k, v in batch.items()}
        b_input_ids = batch["input_ids"].to(device)
        b_input_mask = batch["attention_mask"].to(device)
        b_labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)
        predictions = torch.argmax(outputs, dim=-1)
        preds.extend(predictions.cpu().detach().tolist())
        trues.extend(b_labels.cpu().detach().tolist())

    print(
        classification_report(
            np.array(trues).flatten(), np.array(preds).flatten(), target_names=names
        )
    )

    cm = confusion_matrix(np.array(trues).flatten(), np.array(preds).flatten())
    df_cm = pd.DataFrame(cm, index=names, columns=names)
    # config plot sizes
    sn.set(font_scale=1)
    sn.heatmap(
        df_cm, annot=True, annot_kws={"size": 8}, cmap="coolwarm", linewidth=0.5, fmt=""
    )
    plt.show()


def initiate():
    train_dataset = SiliconeDataset(name="train")
    eval_dataset = SiliconeDataset(name="validation")
    test_dataset = SiliconeDataset(name="test")

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(eval_dataset, batch_size=8)
    test_dataloader = DataLoader(test_dataset, batch_size=8)

    # Load pre-trained BERT model
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    embedding_dim = bert_model.config.hidden_size

    names = train_dataset.labels
    return (
        bert_model,
        embedding_dim,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        names,
    )


def main():
    (
        bert_model,
        embedding_dim,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        names,
    ) = initiate()

    # Create an instance of the classifier
    num_classes = len(names)  # Number of output classes
    model = BertClassifier(bert_model, num_classes, embedding_dim)

    if Path("intent_classification/models/bert_classifier_weights.pth").exists():
        model.load_state_dict(
            torch.load("intent_classification/models/bert_classifier_weights.pth")
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    nb_epochs = 1
    total_loss, total_accuracy = train(
        model, nb_epochs, train_dataloader, eval_dataloader, device, names
    )

    fig, ax1 = plt.subplots()
    ax1.plot(range(len(total_loss)), total_loss, label="loss")
    ax2 = ax1.twinx()
    ax2.plot(range(len(total_loss)), total_accuracy, label="acc")
    fig.tight_layout()
    plt.show()

    test(model, test_dataloader, device, names)


if __name__ == "__main__":
    main()
