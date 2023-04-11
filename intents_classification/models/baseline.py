import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, llm_trained_model, num_classes, embedding_dim):
        super(BaselineModel, self).__init__()
        self.llm_trained_model = llm_trained_model
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, num_classes),
            nn.Dropout(0.15),
            nn.Softmax(dim=0)
        )

        for param in self.llm_trained_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        output = self.llm_trained_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output =  output.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        return logits