from pathlib import Path

from intents_classification.model import Model
from intents_classification.models import (BertClassifier, RobertaClassifier,
                                           SimpleBertClassifier)


def main():
    model = Model(
        dataset_name="dyda_da",
        tokenizer_name="bert-base-uncased",
        language_model_name="bert-base-uncased",
        model_type=BertClassifier,
        model_name="bert_classifier",
        model_dir=Path("models/Bert/saved_models"),
        batch_size=8,
        loss="cross_entropy",
        load=True,
    )

    model.train(nb_epoch=1)
    model.test()


if __name__ == "__main__":
    main()
