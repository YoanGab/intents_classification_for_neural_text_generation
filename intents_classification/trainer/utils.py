from enum import Enum
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np


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
    signals: list[Union[list, np.ndarray]],
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
    if len(signals) > 2:
        ax.plot(range(len(signals[2])), signals[2], label=labels[2], color="green")

    if epochs:
        ax.set_xlabel("Epochs")
    else:
        ax.set_xlabel("Steps")

    ax.set_ylabel(data_type.value)

    fig_title = f"{fig_type.value.replace('_', ' ')} {data_type.value} evolution over {len(signals[0])} {'epochs' if epochs else 'all training'}"

    ax.set_title(fig_title)
    fig.savefig(
        figure_path
        / f"{fig_type.name}_{data_type.value}_{len(signals[0])}_{'epochs' if epochs else 'all'}.png",
        format="png",
    )
