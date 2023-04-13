from pathlib import Path
from transformers import AutoModel

from intents_classification.trainer import Trainer, DatasetUploader, ModelLoader
from intents_classification.models import (BaselineModel, 
                                           ThreeLayersBertClassifier, 
                                           FourLayersBertClassifier)

def main():
    dataset_names = ["meld_s", "iemo","sem", "meld_e", "dyda_e"]
    llm_model_names = ["roberta-base", "bert-base-uncased"]
    classification_models = [BaselineModel, BaselineModel, ThreeLayersBertClassifier, FourLayersBertClassifier]
    classification_model_names = ["baseline", "baseline_fine_tuned", "three_layers", "four_layers"]

    print("Training models...")
    print("========================================")
    for dataset_name in dataset_names:
        print(f"\n\nDataset: {dataset_name}")
        print("----------------------------------------\n")
        for llm_model_name in llm_model_names:
            print(f"\n\nLanguage model: {llm_model_name}")
            print("----------------------------------------\n")
            dataset_loader = DatasetUploader(
                dataset_name=dataset_name,
                tokenizer_name=llm_model_name,
                batch_size=8,
            )

            for classification_model_name, classification_model in zip(classification_model_names, classification_models):
                print(f"\n\nClassification model: {classification_model_name}")
                print("----------------------------------------\n")
                model_loader = ModelLoader(
                    datasetLoader=dataset_loader,
                    language_model_name=llm_model_name,
                    model_type=classification_model,
                    model_name=f"{classification_model_name}_{llm_model_name.split('-')[0]}_{dataset_name}",
                    language_model_class = AutoModel,
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

if __name__ == "__main__":
    main()
