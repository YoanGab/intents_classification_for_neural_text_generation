import numpy as np
import torch

from intents_classification.loaders import ModelLoader
from intents_classification.datasets import CustomData

class Tester:
    def __init__(self, modelLoader: ModelLoader):
        self.modelLoader = modelLoader

    def predict(self, test_sentence: str = None):
        custom_data = CustomData(test_sentence, self.modelLoader.language_model_name)
        input_ids = torch.tensor([custom_data.tokenized_data["input_ids"]]).to(self.modelLoader.device)
        input_mask = torch.tensor([custom_data.tokenized_data["attention_mask"]]).to(self.modelLoader.device)

        with torch.no_grad():
            outputs = self.modelLoader.model(
                input_ids=input_ids,
                attention_mask=input_mask
            )
            predictions = np.argmax(outputs.detach().cpu().numpy(), axis=1)

        pred = self.modelLoader.datasetLoader.unique_labels[predictions[0]]
        return pred 