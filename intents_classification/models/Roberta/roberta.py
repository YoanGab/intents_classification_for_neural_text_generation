import torch.nn as nn


class RobertaClassifier(nn.Module):
    def __init__(self, roberta_model, num_classes, embedding_dim):
        super(RobertaClassifier, self).__init__()
        self.roberta_model = roberta_model
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, num_classes), nn.Softmax(dim=0)
        )

    def forward(self, input_ids, attention_mask):
        output = self.roberta_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output[1]  # Use the output of [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
