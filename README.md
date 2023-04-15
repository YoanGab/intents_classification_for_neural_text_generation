# MLP Architectures for Emotion and Sentiment Analysis: A Comparative Study of Fine-Tuned BERT and RoBERTa Models

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
    python -m intents_classification --mode test --model_path path/to/model.pkl
  ```

## Experimental results
|         |                         | Average | $DyDA_e$ | $IEMO$ | $MELD_e$ | $MELD_s$ | $SEM$ |
|---------|-------------------------|---------|----------|--------|----------|----------|-------|
| BERT    | Baseline                | 0.231   | 0.204    | 0.095  | 0.182    | 0.324    | 0.350 |
|         | Fine-tuned Baseline     | 0.386   | 0.420    | 0.170  | 0.399    | 0.518    | 0.423 |
|         | Three Layers Perceptron | 0.551   | 0.734    | 0.249  | 0.545    | 0.643    | 0.585 |
|         | Four Layers Perceptron  | 0.448   | 0.542    | 0.195  | 0.443    | 0.581    | 0.479 |
| RoBERTa | Baseline                | 0.263   | 0.355    | 0.121  | 0.191    | 0.324    | 0.325 |
|         | Fine-tuned Baseline     | 0.297   | 0.406    | 0.130  | 0.193    | 0.391    | 0.367 |
|         | Three Layers Perceptron | 0.435   | 0.734    | 0.108  | 0.313    | 0.527    | 0.495 |
|         | Four Layers Perceptron  | 0.378   | 0.470    | 0.100  | 0.369    | 0.491    | 0.458 |


## References:

[1] Emile Chapuis,Pierre Colombo, Matthieu Labeau, and Chloé Clavel. Code-switched inspired losses for generic spoken
dialog representations. EMNLP 2021

[2] Emile Chapuis,Pierre Colombo, Matteo Manica, Matthieu Labeau, and Chloé Clavel. Hierarchical pre-training for
sequence labelling in spoken dialog. Finding of EMNLP 2020

[3]Tanvi Dinkar, Pierre Colombo , Matthieu Labeau, and Chloé Clavel. The importance of fillers for text representations
of speech transcripts. EMNLP 2020

[4] Hamid Jalalzai, Pierre Colombo , Chloe Clavel, Eric Gaussier, Giovanna Varni, Emmanuel Vignon, and Anne Sabourin.
Heavy-tailed representations, text polarity classification & data augmentation. NeurIPS 2020

[5] Pierre Colombo, Emile Chapuis, Matteo Manica, Emmanuel Vignon, Giovanna Varni, and Chloé Clavel. Guiding attention
in sequence-to-sequence models for dialogue act prediction. (oral) AAAI 2020

[6] Alexandre Garcia,Pierre Colombo, Slim Essid, Florence d’Alché-Buc, and Chloé Clavel. From the token to the review: A
hierarchical multimodal approach to opinion mining. EMNLP 2020

[7] Pierre Colombo, Wojciech Witon, Ashutosh Modi, James Kennedy, and Mubbasir Kapadia. Affect-driven dialog generation.
NAACL 2019

