import argparse
from pathlib import Path

from transformers import AutoModel

from intents_classification.loaders import DatasetLoader, ModelLoader
from intents_classification.models import (BaselineModel,
                                           FourLayersBertClassifier,
                                           ThreeLayersBertClassifier)
from intents_classification.tester import Tester
from intents_classification.trainer import Trainer


def train_all():
    dataset_names = ["meld_s", "iemo", "sem", "meld_e", "dyda_e"]
    llm_model_names = ["roberta-base", "bert-base-uncased"]
    classification_models = [
        BaselineModel,
        BaselineModel,
        ThreeLayersBertClassifier,
        FourLayersBertClassifier,
    ]
    classification_model_names = [
        "baseline",
        "baseline_fine_tuned",
        "three_layers",
        "four_layers",
    ]

    print("Training models...")
    print("========================================")
    for dataset_name in dataset_names:
        print(f"\n\nDataset: {dataset_name}")
        print("----------------------------------------\n")
        for llm_model_name in llm_model_names:
            print(f"\n\nLanguage model: {llm_model_name}")
            print("----------------------------------------\n")
            dataset_loader = DatasetLoader(
                dataset_name=dataset_name,
                tokenizer_name=llm_model_name,
                batch_size=8,
            )

            for classification_model_name, classification_model in zip(
                classification_model_names, classification_models
            ):
                print(f"\n\nClassification model: {classification_model_name}")
                print("----------------------------------------\n")
                model_loader = ModelLoader(
                    datasetLoader=dataset_loader,
                    language_model_name=llm_model_name,
                    model_type=classification_model,
                    model_name=f"{classification_model_name}_{llm_model_name.split('-')[0]}_{dataset_name}",
                    language_model_class=AutoModel,
                    loss="cross_entropy",
                    load=False,
                )

                model_trainer = Trainer(
                    datasetLoader=dataset_loader,
                    modelLoader=model_loader,
                )

                if dataset_name == "dyda_e":
                    nb_epoch = 1
                if dataset_name == "sem":
                    nb_epoch = 10
                else:
                    nb_epoch = 5

                if classification_model_name != "baseline":
                    model_trainer.train(
                        nb_epoch=nb_epoch,
                        learning_rate=2e-5,
                        clip_grad_norm=1.0,
                        save_freq=500,
                        warm_start=True,
                        scheduler=False,
                    )
                model_trainer.test()


def main():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--mode",
        type=str,
        help="Mode",
        required=True,
        choices=["train", "test", "train_all"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name",
        choices=["meld_s", "iemo", "sem", "meld_e", "dyda_e"],
        default="sem",
    )
    parser.add_argument(
        "--language_model",
        type=str,
        help="Language model name",
        choices=["roberta-base", "bert-base-uncased"],
        default="bert-base-uncased",
    )
    parser.add_argument(
        "--classification_model",
        type=str,
        help="Classification model name",
        choices=["baseline_fine_tuned", "three_layers", "four_layers"],
        default="baseline",
    )
    parser.add_argument("--nb_epoch", type=int, help="Number of epochs", default=5)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=8)
    parser.add_argument(
        "--learning_rate", type=float, help="Learning rate", default=2e-5
    )
    parser.add_argument(
        "--clip_grad_norm", type=float, help="Clip gradient norm", default=1.0
    )
    parser.add_argument("--save", type=bool, help="Save", default=True)
    parser.add_argument("--save_freq", type=int, help="Save frequency", default=500)
    parser.add_argument("--warm_start", type=bool, help="Warm start", default=True)
    parser.add_argument("--scheduler", type=bool, help="Scheduler", default=False)
    parser.add_argument("--checkpoints", type=bool, help="Checkpoints", default=False)
    parser.add_argument(
        "--model_path", type=str, help="Path to the model", default=None
    )
    args = parser.parse_args()

    if args.mode == "train_all":
        train_all()

    dataset_name = args.dataset
    llm_model_name = args.language_model
    classification_model_name = args.classification_model
    nb_epoch = args.nb_epoch
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    clip_grad_norm = args.clip_grad_norm
    save = args.save
    save_freq = args.save_freq
    warm_start = args.warm_start
    scheduler = args.scheduler
    checkpoints = args.checkpoints
    if args.model_path is not None:
        model_path = Path(args.model_path)

    if args.mode == "train":
        classification_model = (
            BaselineModel
            if classification_model_name == "baseline_fine_tuned"
            else ThreeLayersBertClassifier
            if classification_model_name == "three_layers"
            else FourLayersBertClassifier
            if classification_model_name == "four_layers"
            else None
        )
        dataset_loader = DatasetLoader(
            dataset_name=dataset_name,
            tokenizer_name=llm_model_name,
            batch_size=batch_size,
        )

        model_loader = ModelLoader(
            datasetLoader=dataset_loader,
            language_model_name=llm_model_name,
            model_type=classification_model,
            model_name=f"{classification_model_name}_{llm_model_name.split('-')[0]}_{dataset_name}",
            language_model_class=AutoModel,
            loss="cross_entropy",
            load=False,
        )

        model_trainer = Trainer(
            datasetLoader=dataset_loader,
            modelLoader=model_loader,
        )

        model_trainer.train(
            nb_epoch=nb_epoch,
            learning_rate=learning_rate,
            clip_grad_norm=clip_grad_norm,
            save_freq=save_freq,
            save=save,
            warm_start=warm_start,
            scheduler=scheduler,
            checkpoints=checkpoints,
        )

        model_trainer.test()

    if args.mode == "test":
        classification_model = (
            BaselineModel
            if classification_model_name == "baseline_fine_tuned"
            else ThreeLayersBertClassifier
            if classification_model_name == "three_layers"
            else FourLayersBertClassifier
            if classification_model_name == "four_layers"
            else BaselineModel
        )
        dataset_loader = DatasetLoader(
            dataset_name=dataset_name,
            tokenizer_name=llm_model_name,
            batch_size=batch_size,
        )

        if classification_model_name == "baseline":
            model_loader = ModelLoader(
                datasetLoader=dataset_loader,
                language_model_name=llm_model_name,
                model_type=classification_model,
                model_name=f"{classification_model_name}_{llm_model_name.split('-')[0]}_{dataset_name}",
                language_model_class=AutoModel,
                loss="cross_entropy",
                load=False,
            )
        else:
            model_loader = ModelLoader(
                datasetLoader=dataset_loader,
                language_model_name=llm_model_name,
                model_type=classification_model,
                model_name=f"{classification_model_name}_{llm_model_name.split('-')[0]}_{dataset_name}",
                language_model_class=AutoModel,
                loss="cross_entropy",
                load=True,
                model_path=model_path,
            )

        model_tester = Tester(modelLoader=model_loader)

        while True:
            sentence = input("Enter a sentence: ")
            if sentence == "exit":
                break
            pred = model_tester.predict(sentence)
            print("Prediction: ", pred)


if __name__ == "__main__":
    main()
