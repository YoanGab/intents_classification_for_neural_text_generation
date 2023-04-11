from pathlib import Path
from transformers import AutoModel

from intents_classification.trainer import Trainer, DatasetUploader, ModelLoader
from intents_classification.models import (BaselineModel, 
                                           ThreeLayersBertClassifier, 
                                           MultiPerceptronBertClassifier)


def main():
    datasetLoader = DatasetUploader(
        dataset_name="dyda_da",
        tokenizer_name="bert-base-uncased",
        batch_size=8,
    )

    modelLoader = ModelLoader(
        datasetLoader=datasetLoader,
        language_model_name="bert-base-uncased",
        model_type=ThreeLayersBertClassifier,
        model_name="bert_classifier",
        model_dir=Path("models/Bert/saved_models"),
        language_model_class = AutoModel,
        loss="cross_entropy",
        load=False,
    )

    trainer = Trainer(
        datasetLoader=datasetLoader,
        modelLoader=modelLoader,
    )

    trainer.train(
        nb_epoch=10,
        learning_rate=2e-5,
        clip_grad_norm=1.0,
        save_freq=1000,
        warm_start=True,
        scheduler=False,
    )

    trainer.test()


if __name__ == "__main__":
    main()
