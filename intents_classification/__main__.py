from pathlib import Path
from transformers import AutoModel

from intents_classification.trainer import Trainer, DatasetUploader, ModelLoader
from intents_classification.models import (BaselineModel, 
                                           ThreeLayersBertClassifier, 
                                           FourLayersBertClassifier)


def main():
    dataset_loader = DatasetUploader(
        dataset_name="sem",
        tokenizer_name="bert-base-uncased",
        batch_size=8,
    )

    baseline_model_loader = ModelLoader(
        datasetLoader=dataset_loader,
        language_model_name="bert-base-uncased",
        model_type=BaselineModel,
        model_name="baseline",
        language_model_class = AutoModel,
        loss="cross_entropy",
        load=False,
    )
    three_layers_model_loader = ModelLoader(
        datasetLoader=dataset_loader,
        language_model_name="bert-base-uncased",
        model_type=ThreeLayersBertClassifier,
        model_name="three_layers_model",
        language_model_class = AutoModel,
        loss="cross_entropy",
        load=False,
    )
    four_layers_model_loader = ModelLoader(
        datasetLoader=dataset_loader,
        language_model_name="bert-base-uncased",
        model_type=FourLayersBertClassifier,
        model_name="four_layers_model",
        language_model_class = AutoModel,
        loss="cross_entropy",
        load=False,
    )

    baseline_model_trainer = Trainer(
        datasetLoader=dataset_loader,
        modelLoader=baseline_model_loader,
    )
    three_layers_model_trainer = Trainer(
        datasetLoader=dataset_loader,
        modelLoader=three_layers_model_loader,
    )
    four_layers_model_trainer = Trainer(
        datasetLoader=dataset_loader,
        modelLoader=four_layers_model_loader,
    )

    baseline_model_trainer.train(
        nb_epoch=6,
        learning_rate=2e-5,
        clip_grad_norm=1.0,
        save_freq=500,
        warm_start=True,
        scheduler=False,
    )
    three_layers_model_trainer.train(
        nb_epoch=6,
        learning_rate=2e-5,
        clip_grad_norm=1.0,
        save_freq=500,
        warm_start=True,
        scheduler=False,
    )
    four_layers_model_trainer.train(
        nb_epoch=6,
        learning_rate=2e-5,
        clip_grad_norm=1.0,
        save_freq=500,
        warm_start=True,
        scheduler=False,
    )

    baseline_model_trainer.test()
    three_layers_model_trainer.train()
    four_layers_model_trainer.train()


if __name__ == "__main__":
    main()
