from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class SiliconeDataset(Dataset):
    def __init__(
        self,
        set_type,
        dataset_name="dyda_da",
        tokenizer_name="bert-base-uncased",
        columns_to_keep=["labels", "input_ids", "attention_mask"],
    ):
        self.columns_to_keep = columns_to_keep
        self.dataset = load_dataset("silicone", dataset_name)[set_type]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self._tokenize_dataset()
        self.labels = self.dataset.features["Label"].names

    def _tokenize_dataset(self):
        self.tokenized_datasets = self.dataset.map(
            lambda example: self.tokenizer(
                example["Utterance"], padding="max_length", truncation=True
            ),
            batched=True,
        )
        self.tokenized_datasets = self.tokenized_datasets.rename_column(
            "Label", "labels"
        )
        columns_to_remove = set(self.tokenized_datasets.features.keys()).difference(
            self.columns_to_keep
        )
        self.tokenized_datasets = self.tokenized_datasets.remove_columns(
            columns_to_remove
        )
        self.tokenized_datasets.set_format("torch")

    def __getitem__(self, idx):
        return self.tokenized_datasets[idx]

    def __len__(self):
        return len(self.tokenized_datasets)
