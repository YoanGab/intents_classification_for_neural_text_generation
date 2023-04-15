from enum import Enum
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class FigureType(Enum):
    TRAIN_VAL = "Train_Validation"
    TRAIN_TEST = "Train_Test"
    VAL_TEST = "Validation_Test"
    TRAIN_VAL_TEST = "Train_Validation_Test"
    TRAIN = "Train"
    VAL = "Validation"
    TEST = "Test"


class DataType(Enum):
    LOSS = "Loss"
    ACCURACY = "Accuracy"


def flat_accuracy(preds: Union[list, np.ndarray], labels: Union[list, np.ndarray]):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def plot_comparison(
    model_name: str,
    signals: list,
    fig_type: FigureType,
    data_type: DataType,
    epochs: bool,
):
    figure_path = Path("intents_classification") / "figs" / model_name
    figure_path.mkdir(parents=True, exist_ok=True)

    labels = fig_type.value.split("_")

    fig, ax = plt.subplots()
    ax.plot(range(len(signals[0])), signals[0], label=labels[0])
    if len(signals) > 1:
        ax.plot(range(len(signals[1])), signals[1], label=labels[1], color="orange")
        ax.legend()
    if len(signals) > 2:
        ax.plot(range(len(signals[2])), signals[2], label=labels[2], color="green")
        ax.legend()

    if epochs:
        ax.set_xlabel("Epochs")
    else:
        ax.set_xlabel("Steps")

    ax.set_ylabel(data_type.value)

    fig_title = f"{fig_type.value.replace('_', ' ')} {data_type.value} evolution over {str(len(signals[0])) + ' epochs' if epochs else 'all training'}"

    ax.set_title(fig_title)
    fig.savefig(
        figure_path
        / f"{fig_type.name}_{data_type.value}_{str(len(signals[0])) + '_epochs' if epochs else 'all'}.png",
        format="png",
    )


def plot_acc_val(model_dir, val_loss_per_epoch, val_accuracy_per_epoch, train_accuracy_per_epoch, train_loss_per_epoch):
    fig, ax = plt.subplots()
    ax.plot(range(len(val_loss_per_epoch)), val_loss_per_epoch, label="Validation Loss", color="orange")
    ax.plot(range(len(train_loss_per_epoch)), train_loss_per_epoch, label="Train Loss", color="green")
    ax2 = ax.twinx()
    ax2.plot(range(len(val_accuracy_per_epoch)), val_accuracy_per_epoch, label="Validation Accuracy", color="red")
    ax2.plot(range(len(train_accuracy_per_epoch)), train_accuracy_per_epoch, label="Train Accuracy", color="blue")
    ax.legend()
    ax2.legend()
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy")
    fig_title = f"Train and Validation Loss and Accuracy evolution over {str(len(val_loss_per_epoch))} epochs"
    ax.set_title(fig_title)
    fig.savefig(
        model_dir / "confusion_matrix.csv" / f"train_val_loss_acc.png", format="png"
    )


def analyze_results():
    results = {}
    model_folders = [x for x in Path("Results").iterdir() if x.is_dir()]
    for model_folder in model_folders:
        llm_folders = [x for x in model_folder.iterdir() if x.is_dir()]
        results[model_folder.name] = {}
        for llm_folder in llm_folders:
            dataset_folders = [x for x in llm_folder.iterdir() if x.is_dir()]
            results[model_folder.name][llm_folder.name] = {}
            for dataset_folder in dataset_folders:
                csv_files = [x for x in dataset_folder.iterdir() if x.suffix == ".csv"]
                for csv_file in csv_files:
                    if csv_file.name == "classification_report.csv":
                        df = pd.read_csv(csv_file, index_col=0)
                        results[model_folder.name][llm_folder.name][
                            dataset_folder.name
                        ] = df

    inverted_results = {}

    for baseline, baseline_data in results.items():
        for model, model_data in baseline_data.items():
            for metric, value in model_data.items():
                if metric not in inverted_results:
                    inverted_results[metric] = {}
                if model not in inverted_results[metric]:
                    inverted_results[metric][model] = {}
                inverted_results[metric][model][baseline] = value

    dataset_results_df = {}
    for dataset_folder in dataset_folders[1:]:
        dataset_folder = dataset_folder.name
        df_dict = {}

        first_col_index = [
            v
            for v in inverted_results[dataset_folder]
            for _ in list(inverted_results[dataset_folder][v].keys()) + [f"best_{v}"]
        ] + ["best"]
        second_col_index = [
            val
            for k in inverted_results[dataset_folder]
            for val in list(inverted_results[dataset_folder][k].keys()) + [f"best_{k}"]
        ] + ["best"]
        arrays = [first_col_index, second_col_index]
        tuples = list(zip(*arrays))
        columns = pd.MultiIndex.from_tuples(tuples)

        for i, key in enumerate(inverted_results[dataset_folder]):
            if i == 0:
                first_row_index = [
                    class_name
                    for class_name in inverted_results[dataset_folder][key][
                        "baseline"
                    ].index
                    for _ in inverted_results[dataset_folder][key]["baseline"].columns[
                        :-1
                    ]
                ] + ["best"]
                second_row_index = [
                    metric
                    for _ in inverted_results[dataset_folder][key]["baseline"].index
                    for metric in inverted_results[dataset_folder][key][
                        "baseline"
                    ].columns[:-1]
                ] + ["best"]

                arrays = [first_row_index, second_row_index]
                tuples = list(zip(*arrays))
                index = pd.MultiIndex.from_tuples(tuples)

            for model_name, df in inverted_results[dataset_folder][key].items():
                values = np.concatenate(
                    (df.iloc[:, :-1].values.flatten(), np.array([0])), axis=0
                )
                df_dict[(key, model_name)] = np.concatenate(
                    (values[:-8], values[-6:]), axis=0
                )

            df_dict[(key, f"best_{key}")] = np.zeros(len(first_row_index) - 2)
        df_dict[("best", "best")] = np.zeros(len(first_row_index) - 2)
        index = index.drop([("accuracy", "recall"), ("accuracy", "f1-score")])

        df = pd.DataFrame(df_dict, index=index, columns=columns)
        # df = df.rename(index={'precision': 'accuracy'}, level=1)
        for key in inverted_results[dataset_folder]:
            df.loc[:, (key, f"best_{key}")] = df.loc[:, key].apply(
                lambda x: df.loc[:, "bert"].columns[:-1][x.argmax()], axis=1
            )
            df.loc["best", (key, f"best_{key}")] = df.loc[
                :, (key, f"best_{key}")
            ].mode()[0]
        df[("best", "best")] = df.apply(
            lambda x: list(inverted_results[dataset_folder].keys())[
                np.argmax([x.iloc[:3].max(), x.iloc[4:-2].max()])
            ],
            axis=1,
        )
        best_llm = df.loc[("best", "best"), ("best", "best")]
        best_model = df[best_llm].loc[("best", "best"), f"best_{best_llm}"]
        df.loc[("best", "best"), ("best", "best")] = best_llm + " " + best_model
        dataset_results_df[dataset_folder] = df

    dataset_results = {}
    for dataset_folder in dataset_folders[1:]:
        dataset_results[dataset_folder.name] = pd.DataFrame(
            dataset_results_df[dataset_folder.name].loc[
                [
                    ("weighted avg", "precision"),
                    ("weighted avg", "recall"),
                    ("weighted avg", "f1-score"),
                ],
                [
                    (llm_name, model_name)
                    for llm_name in inverted_results[dataset_folder.name].keys()
                    for model_name in inverted_results[dataset_folder.name][
                        llm_name
                    ].keys()
                ],
            ]
        )

    return dataset_results, dataset_results_df
