from transformers import AutoTokenizer

class CustomData:
    def __init__(self, 
                 data: str,
                 tokenizer_name="bert-base-uncased"):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self._tokenize_dataset()
    
    def _tokenize_dataset(self):
        self.tokenized_data = self.tokenizer(
            self.data, padding="max_length", truncation=True
        )