from torch.utils.data import DataLoader

from intents_classification.datasets import SiliconeDataset


class DatasetLoader:
    def __init__(self, dataset_name: str, tokenizer_name: str, batch_size=8):
        self._load_datasets(dataset_name, tokenizer_name)
        self._load_dataloader(batch_size)

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
