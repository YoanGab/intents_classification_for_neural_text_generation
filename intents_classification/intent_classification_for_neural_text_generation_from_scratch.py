# -*- coding: utf-8 -*-
#!pip install datasets
#!pip install transformers

from datasets import load_dataset, load_metric
import time

import torch.nn as nn
from transformers import BertModel, BertTokenizer

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from tqdm.auto import tqdm

def predict(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    return names[predictions[0].item()]

dataset = load_dataset("silicone", "dyda_da")
names = dataset["train"].features["Label"].names

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
tokenized_datasets = dataset.map(lambda example: tokenizer(example["Utterance"], padding="max_length", truncation=True), batched=True)

tokenized_datasets = tokenized_datasets.remove_columns(["Dialogue_ID", "Dialogue_Act", "Idx", "Utterance", "token_type_ids"])
tokenized_datasets = tokenized_datasets.rename_column("Label", "labels")
tokenized_datasets.set_format("torch")

train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=8)
eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=8)
test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=8)

# Load pre-trained BERT model
bert_model = BertModel.from_pretrained('bert-base-uncased')
embedding_dim = bert_model.config.hidden_size

# Add a multi-layer neural network on top of BERT
class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output[1]  # Use the output of [CLS] token 
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Create an instance of the classifier
num_classes = len(names)  # Number of output classes
model = BertClassifier(bert_model, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=5e-5)

# Train the model
def train(model, train_dataloader, optimizer, criterion, device):
    train_loss = 0
    progress_bar = tqdm(range(len(train_dataloader)))
    model.train()
    for step, batch in enumerate(train_dataloader):
        # Load batch to GPU
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)

        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(b_input_ids, attention_mask=b_input_mask)
        loss, logits = outputs[:2]
        loss = loss.mean()

        # Backward pass
        loss.backward()

        # Update parameters and take a step using the computed gradient
        optimizer.step()

        # Update tracking variables
        train_loss += loss.item()
        
        progress_bar.set_description(f'Training loss: {round(train_loss/(step+1),3)}')
        progress_bar.update(1)
    train_mean_loss = train_loss/len(train_dataloader)
    return train_mean_loss

def eval(model, eval_dataloader, device):    
    val_loss = 0
    nb_val_steps = 0
    nb_val_correct = 0
    nb_val_total = 0
    progress_bar = tqdm(range(len(eval_dataloader)))
    model.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            # Load batch to GPU
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            loss, logits = outputs[:2]

            # Calculate mean loss
            loss = loss.mean()
            val_loss += loss.item()
            nb_val_steps += 1

            # Calculate accuracy
            preds = torch.argmax(logits)
            nb_val_correct += (preds == b_labels).sum().item()
            nb_val_total += len(b_labels)
            progress_bar.update(1)

    val_mean_loss = val_loss / nb_val_steps
    val_mean_accuracy = nb_val_correct / nb_val_total
    return val_mean_loss, val_mean_accuracy

# Define the training parameters
epochs = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train the model for multiple epochs
for epoch in range(epochs):
    print("EPOCH", epoch)
    print("==========================TRAINING==========================")
    train_loss = train(model, train_dataloader, optimizer, criterion, device)
    print("==========================EVALUATION==========================")
    val_loss, val_accuracy = eval(model, eval_dataloader, device)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {np.mean(train_loss):.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

# Save model
model.save_pretrained('models/bert')

"""
metric = load_metric("accuracy")
model.eval()
preds, trues = [], []
for i, batch in tqdm(enumerate(test_dataloader), desc="evaluating", total=test_dataloader.__len__()):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

    _, tag_seq  = torch.max(logits, 1)
    preds.extend(tag_seq.cpu().detach().tolist())
    trues.extend(batch['labels'].cpu().detach().tolist())

metric.compute()

print(classification_report(np.array(trues).flatten(), np.array(preds).flatten(), target_names=names))

cm = confusion_matrix(np.array(trues).flatten(), np.array(preds).flatten())
df_cm = pd.DataFrame(cm, index=names, columns=names)
# config plot sizes
sn.set(font_scale=1)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, cmap='coolwarm', linewidth=0.5, fmt="")
plt.show()

"""

