import torch.nn as nn

from intents_classification.models.baseline import BaselineModel

class ThreeLayersBertClassifier(BaselineModel, nn.Module):
    def __init__(self, llm_trained_model, num_classes, embedding_dim):
        super(ThreeLayersBertClassifier, self).__init__(
            llm_trained_model, num_classes, embedding_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(768, num_classes),
            nn.Softmax(dim=0),
        )


class FourLayersBertClassifier(BaselineModel, nn.Module):
    def __init__(self, llm_trained_model, num_classes, embedding_dim):
        super(FourLayersBertClassifier, self).__init__(
            llm_trained_model, num_classes, embedding_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(768, num_classes),
            nn.Softmax(dim=1),
        )
