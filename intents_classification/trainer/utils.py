from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)

def plot_acc_loss(model_name, total_loss, total_accuracy):
    figure_path = Path("intents_classification/figs") / model_name
    figure_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(range(len(total_loss)), total_loss)
    ax.set_title(f"Evolution of loss over {len(total_loss)} epochs")
    fig.savefig(figure_path / f"loss.png", format="png")
    fig, ax = plt.subplots()
    ax.plot(range(len(total_accuracy)), total_accuracy)
    ax.set_title(f"Evolution of accuracy over {len(total_accuracy)} epochs")
    fig.savefig(figure_path / f"acc.png", format="png")