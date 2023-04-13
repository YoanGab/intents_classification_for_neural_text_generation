# MLP Architectures for Emotion and Sentiment Analysis: A Comparative Study of Fine-Tuned BERT and RoBERTa Models

This project 

## Abstract
> This paper presents a comparative study of multi-layer perceptron (MLP) architectures for E/S analysis using fine-tuned BERT and RoBERTa models. We evaluate the performance of different MLP architectures on three datasets from Sequence labellIng evaLuatIon benChmark fOr spoken laNguagE (SILICONE): SEM, dyda\_e, and meld\_e. Specifically, we compare the baseline models (BERT and RoBERTa without fine-tuning) with fine-tuned models and two MLP architectures, namely a three-layer MLP and a four-layer MLP, both fine-tuned with BERT and RoBERTa. Our results show that fine-tuning significantly improves the performance of the baseline models, and MLP architectures outperform the baseline models in terms of accuracy. Furthermore, we observe that the four-layer MLP outperforms the three-layer MLP, and RoBERTa generally outperforms BERT across all architectures. Overall, our study provides insights into the effectiveness of different MLP architectures for E/S analysis, and highlights the benefits of fine-tuning pre-trained language models for this task.

## Getting start
1. Clone the repository:
```bash
    git clone https://github.com/YoanGab/intents_classification_for_neural_text_generation.git
```  
2. Download dependencies (prefered in a virtual env)
```bash
    pip install --upgrade pip
    pip install -r requirements.txt
```
3. Run the package
```bash
    cd intents_classification_for_neural_text_generation
```
  a. Fine-tuned a model on a dataset
  ```bash
    python -m intents_classification --mode train
  ```
  b. Test a fine-tuned model
  ```bash
    python -m intents_classification --mode test
  ```
  
## Datasets
The datasets used in this project is silicone from Huggingface datasets : https://huggingface.co/datasets/silicone

## Models



## License
All source code is made available under a MIT license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE` for the full license text.
