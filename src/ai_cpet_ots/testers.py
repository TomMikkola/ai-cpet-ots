"""Evaluate published models over complete loaders and optionally plot results.

All evaluators return (AUROC, AUPRC, F1, accuracy) tensors and use the model
threshold for F1 and accuracy. Ensemble evaluators can additionally return
probabilities and targets in loader order. Callers must preserve subject order
across modalities. Evaluation runs without gradients on the model device."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix
from torch.nn.utils.rnn import unpad_sequence
from torcheval.metrics.functional import (
    binary_accuracy,
    binary_auprc,
    binary_auroc,
    binary_f1_score,
)

from ai_cpet_ots.paths import output_path


def test_model_1(
    model, model_name, loader, plot_cm=False, plot_PRC=False, plot_predictions=False, plot_ROC=False
):
    """Evaluate model-1 over all loader batches.

    Returns AUROC, AUPRC, F1, and accuracy; plot flags select diagnostic panels.
    Uses model.threshold, keeps loader order, and disables gradient tracking."""
    device = next(model.parameters()).device
    model.eval()
    prediction_batches, target_batches = [], []
    with torch.no_grad():
        for batch in loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.float().to(device), y_batch.to(device)

            y_pred, _ = model(x_batch)
            y_pred = F.sigmoid(y_pred).flatten()

            prediction_batches.append(y_pred)
            target_batches.append(y_batch)

    if not prediction_batches:
        raise ValueError("Evaluation loader must contain at least one batch.")
    y_pred = torch.cat(prediction_batches).flatten()
    y_batch = torch.cat(target_batches).flatten()
    auroc = binary_auroc(y_pred, y_batch)
    auprc = binary_auprc(y_pred, y_batch)
    f1 = binary_f1_score(y_pred, y_batch, threshold=model.threshold)
    acc = binary_accuracy(y_pred, y_batch, threshold=model.threshold)

    # Compute avg AUCPR, F1, ACC over varying thresholds
    th = np.linspace(0, 1, 100)
    th_list = []
    for t in th:
        th_list.append(binary_f1_score(y_pred, y_batch, threshold=t).cpu())

    n_plots = sum([plot_cm, plot_PRC, plot_predictions, plot_ROC])

    if n_plots >= 1:
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        axs = axs.flatten()
        pred_plotted, PRC_plotted, cm_plotted, ROC_plotted = False, False, False, False

        for i in range(n_plots):
            ax = axs[i]

            if plot_predictions and not pred_plotted:
                ax.plot(y_pred.cpu(), "o", label="Predicted")
                ax.hlines(
                    model.threshold,
                    0,
                    len(y_pred),
                    "r",
                    linestyles="dashed",
                    label="Decision threshold",
                )
                ax.set_ylim((-0.05, 1.05))
                ax.set_title("Predictions", fontsize=14)
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_ylabel("Probability", fontsize=14)
                ax.tick_params(axis="both", labelsize=12)
                ax.legend()
                pred_plotted = True
                continue

            if plot_cm and not cm_plotted:
                # Plot confusion matrix
                y_preds = [1 if val >= model.threshold else 0 for val in y_pred]
                cm = confusion_matrix(y_batch.cpu(), y_preds)
                TN = cm[0, 0]
                FP = cm[0, 1]
                FN = cm[1, 0]
                TP = cm[1, 1]

                sensitivity = TP / (TP + FN)
                specificity = TN / (TN + FP)
                ppv = TP / (TP + FP)
                npv = TN / (TN + FN)

                cm = np.vstack([cm, np.array([npv, ppv])])
                cm = np.hstack([cm, np.array([specificity, sensitivity, 0]).reshape(3, 1)])

                labels = np.array(
                    [
                        [int(cm[0, 0]), int(cm[0, 1]), f"Specificity\n{cm[0, 2]:.2f}"],
                        [int(cm[1, 0]), int(cm[1, 1]), f"Sensitivity\n{cm[1, 2]:.2f}"],
                        [f"NPV\n{cm[2, 0]:.2f}", f"PPV\n{cm[2, 1]:.2f}", ""],
                    ]
                )

                mask = np.zeros_like(cm, dtype=bool)
                mask[2, 2] = True

                sns.heatmap(
                    cm,
                    annot=labels,
                    fmt="",
                    cmap="Blues",
                    mask=mask,
                    xticklabels=["Control", "OTS"],
                    yticklabels=["Control", "OTS"],
                    annot_kws={"size": 12},
                    ax=ax,
                )
                ax.set_ylabel("True labels", fontsize=14)
                ax.set_xlabel("Predicted labels", fontsize=14)
                ax.set_title("Confusion matrix", fontsize=14)
                ax.tick_params(axis="both", labelsize=12)
                cm_plotted = True
                continue

            if plot_ROC and not ROC_plotted:
                # Plot ROC-curve
                display = RocCurveDisplay.from_predictions(
                    y_batch.cpu().numpy(), y_pred.cpu(), plot_chance_level=True, ax=ax
                )
                _ = display.ax_.set_title("ROC curve", fontsize=14)
                display.ax_.set_ylabel("Sensitivity", fontsize=14)
                display.ax_.set_xlabel("1-Specificity", fontsize=14)
                ax.tick_params(axis="both", labelsize=12)
                ROC_plotted = True
                continue

            if plot_PRC and not PRC_plotted:
                # Plot PR-curve
                display = PrecisionRecallDisplay.from_predictions(
                    y_batch.cpu().numpy(), y_pred.cpu(), plot_chance_level=True, ax=ax
                )
                _ = display.ax_.set_title("PR curve", fontsize=14)
                display.ax_.set_ylabel("Precision", fontsize=14)
                display.ax_.set_xlabel("Recall", fontsize=14)
                ax.tick_params(axis="both", labelsize=12)
                PRC_plotted = True
                continue
        fig.suptitle(f"{model_name} (Max F1 ({np.max(th_list):.3f})", fontsize=14)

    # ablation_results[model_name] = np.mean([np.max(th_list), auprc.cpu()])
    if n_plots != 0:
        plt.tight_layout()
        plt.show()

    return auroc, auprc, f1, acc


def test_model_2(
    model, model_name, loader, plot_cm=False, plot_PRC=False, plot_predictions=False, plot_ROC=False
):
    """Evaluate model-2 over all loader batches.

    Returns AUROC, AUPRC, F1, and accuracy; plot flags select diagnostic panels.
    Uses model.threshold, keeps loader order, and disables gradient tracking."""
    device = next(model.parameters()).device
    model.eval()
    prediction_batches, target_batches = [], []
    with torch.no_grad():
        for batch in loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.float().to(device), y_batch.to(device)

            y_pred, _ = model(x_batch)
            y_pred = F.sigmoid(y_pred).flatten()

            prediction_batches.append(y_pred)
            target_batches.append(y_batch)

    if not prediction_batches:
        raise ValueError("Evaluation loader must contain at least one batch.")
    y_pred = torch.cat(prediction_batches).flatten()
    y_batch = torch.cat(target_batches).flatten()
    auroc = binary_auroc(y_pred, y_batch)
    auprc = binary_auprc(y_pred, y_batch)
    f1 = binary_f1_score(y_pred, y_batch, threshold=model.threshold)
    acc = binary_accuracy(y_pred, y_batch, threshold=model.threshold)

    # Compute avg AUCPR, F1, ACC over varying thresholds
    th = np.linspace(0, 1, 100)
    th_list = []
    for t in th:
        th_list.append(binary_f1_score(y_pred, y_batch, threshold=t).cpu())

    n_plots = sum([plot_cm, plot_PRC, plot_predictions, plot_ROC])

    if n_plots >= 1:
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        axs = axs.flatten()
        pred_plotted, PRC_plotted, cm_plotted, ROC_plotted = False, False, False, False

        for i in range(n_plots):
            ax = axs[i]

            if plot_predictions and not pred_plotted:
                ax.plot(y_pred.cpu(), "o", label="Predicted")
                ax.hlines(
                    model.threshold,
                    0,
                    len(y_pred),
                    "r",
                    linestyles="dashed",
                    label="Decision threshold",
                )
                ax.set_ylim((-0.05, 1.05))
                ax.set_title("Predictions", fontsize=14)
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_ylabel("Probability", fontsize=14)
                ax.tick_params(axis="both", labelsize=12)
                ax.legend()
                pred_plotted = True
                continue

            if plot_cm and not cm_plotted:
                # Plot confusion matrix
                y_preds = [1 if val >= model.threshold else 0 for val in y_pred]
                cm = confusion_matrix(y_batch.cpu(), y_preds)
                TN = cm[0, 0]
                FP = cm[0, 1]
                FN = cm[1, 0]
                TP = cm[1, 1]

                sensitivity = TP / (TP + FN)
                specificity = TN / (TN + FP)
                ppv = TP / (TP + FP)
                npv = TN / (TN + FN)

                cm = np.vstack([cm, np.array([npv, ppv])])
                cm = np.hstack([cm, np.array([specificity, sensitivity, 0]).reshape(3, 1)])

                labels = np.array(
                    [
                        [int(cm[0, 0]), int(cm[0, 1]), f"Specificity\n{cm[0, 2]:.2f}"],
                        [int(cm[1, 0]), int(cm[1, 1]), f"Sensitivity\n{cm[1, 2]:.2f}"],
                        [f"NPV\n{cm[2, 0]:.2f}", f"PPV\n{cm[2, 1]:.2f}", ""],
                    ]
                )

                mask = np.zeros_like(cm, dtype=bool)
                mask[2, 2] = True

                sns.heatmap(
                    cm,
                    annot=labels,
                    fmt="",
                    cmap="Blues",
                    mask=mask,
                    xticklabels=["Control", "OTS"],
                    yticklabels=["Control", "OTS"],
                    annot_kws={"size": 12},
                    ax=ax,
                )
                ax.set_ylabel("True labels", fontsize=14)
                ax.set_xlabel("Predicted labels", fontsize=14)
                ax.set_title("Confusion matrix", fontsize=14)
                ax.tick_params(axis="both", labelsize=12)
                cm_plotted = True
                continue

            if plot_ROC and not ROC_plotted:
                # Plot ROC-curve
                display = RocCurveDisplay.from_predictions(
                    y_batch.cpu().numpy(), y_pred.cpu(), plot_chance_level=True, ax=ax
                )
                _ = display.ax_.set_title("ROC", fontsize=14)
                display.ax_.set_ylabel("Sensitivity", fontsize=14)
                display.ax_.set_xlabel("1-Specificity", fontsize=14)
                ax.tick_params(axis="both", labelsize=12)
                ROC_plotted = True
                continue

            if plot_PRC and not PRC_plotted:
                # Plot PR-curve
                display = PrecisionRecallDisplay.from_predictions(
                    y_batch.cpu().numpy(), y_pred.cpu(), plot_chance_level=True, ax=ax
                )
                _ = display.ax_.set_title("PR curve", fontsize=14)
                display.ax_.set_ylabel("Precision", fontsize=14)
                display.ax_.set_xlabel("Recall", fontsize=14)
                ax.tick_params(axis="both", labelsize=12)
                PRC_plotted = True
                continue
        fig.suptitle(f"{model_name} (Max F1 ({np.max(th_list):.3f})", fontsize=14)

    # ablation_results[model_name] = np.mean([np.max(th_list), auprc.cpu()])
    if n_plots != 0:
        plt.tight_layout()
        plt.show()

    return auroc, auprc, f1, acc


def test_model_3(
    model,
    model_name,
    loader,
    window_size,
    overlap,
    plot_cm=False,
    plot_PRC=False,
    plot_predictions=False,
    plot_ROC=False,
):
    """Evaluate model-3 over all loader batches.

    Returns AUROC, AUPRC, F1, and accuracy; plot flags select diagnostic panels.
    Uses model.threshold, keeps loader order, and disables gradient tracking."""
    device = next(model.parameters()).device
    model.eval()

    y_preds = torch.empty(0).to(device)
    y_batches = torch.empty(0).to(device)

    with torch.no_grad():
        for batch in loader:
            x_batch, y_batch, seq_lengths = batch
            x_batch, y_batch = x_batch.float().to(device), torch.as_tensor(y_batch, device=device)

            x_batch = unpad_sequence(x_batch, torch.as_tensor(seq_lengths), batch_first=True)

            for i, x in enumerate(x_batch):
                x_splitted = x.unfold(dimension=0, size=window_size, step=overlap).to(device)

                # Add padding to last window if not equal to window_Size
                n_windows = x.size(0) // (window_size - overlap)
                if x_splitted.size(0) < n_windows:
                    last_idx = int(
                        (x_splitted.size(0) * x_splitted.size(-1)) // (window_size / overlap)
                    )
                    padding = torch.zeros((1, 14, window_size - (x.size(0) - last_idx))).to(device)
                    last_data = x[last_idx:].permute(1, 0).unsqueeze(0)
                    last_win = torch.cat((last_data, padding), dim=2)
                    x_splitted = torch.cat((x_splitted, last_win), dim=0)

                x_splitted = x_splitted.float()

                y_pred, _, _, _ = model(x_splitted)

                # Aggregate batch predictions
                y_pred = F.sigmoid(y_pred)
                y_preds = torch.cat((y_preds, y_pred.mean(dim=0)), dim=0)

            # Aggregate ground truths
            y_batches = torch.cat((y_batches, y_batch), dim=0)

    y_preds, y_batches = y_preds.flatten(), y_batches.flatten()

    auroc = binary_auroc(y_preds, y_batches)
    auprc = binary_auprc(y_preds, y_batches)
    f1 = binary_f1_score(y_preds, y_batches, threshold=model.threshold)
    acc = binary_accuracy(y_preds, y_batches, threshold=model.threshold)

    # Compute avg F1, ACC over varying thresholds
    th = np.linspace(0, 1, 100)
    th_list = []
    for t in th:
        th_list.append(binary_f1_score(y_preds, y_batches, threshold=t).cpu())

    n_plots = sum([plot_cm, plot_PRC, plot_predictions, plot_ROC])

    if n_plots >= 1:
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        axs = axs.flatten()
        pred_plotted, PRC_plotted, cm_plotted, ROC_plotted = False, False, False, False

        for i in range(n_plots):
            ax = axs[i]

            if plot_predictions and not pred_plotted:
                ax.plot(y_preds.cpu(), "o", label="Predicted")
                ax.hlines(
                    model.threshold,
                    0,
                    len(y_preds),
                    "r",
                    linestyles="dashed",
                    label="Decision threshold",
                )
                ax.set_ylim((-0.05, 1.05))
                ax.set_title("Predictions", fontsize=14)
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_ylabel("Probability", fontsize=14)
                ax.tick_params(axis="both", labelsize=12)
                ax.legend()
                pred_plotted = True
                continue

            if plot_cm and not cm_plotted:
                # Plot confusion matrix
                y_pred = [1 if val >= model.threshold else 0 for val in y_preds]
                cm = confusion_matrix(y_batches.cpu(), y_pred)
                TN = cm[0, 0]
                FP = cm[0, 1]
                FN = cm[1, 0]
                TP = cm[1, 1]

                sensitivity = TP / (TP + FN)
                specificity = TN / (TN + FP)
                ppv = TP / (TP + FP)
                npv = TN / (TN + FN)

                cm = np.vstack([cm, np.array([npv, ppv])])
                cm = np.hstack([cm, np.array([specificity, sensitivity, 0]).reshape(3, 1)])

                labels = np.array(
                    [
                        [int(cm[0, 0]), int(cm[0, 1]), f"Specificity\n{cm[0, 2]:.2f}"],
                        [int(cm[1, 0]), int(cm[1, 1]), f"Sensitivity\n{cm[1, 2]:.2f}"],
                        [f"NPV\n{cm[2, 0]:.2f}", f"PPV\n{cm[2, 1]:.2f}", ""],
                    ]
                )

                mask = np.zeros_like(cm, dtype=bool)
                mask[2, 2] = True

                sns.heatmap(
                    cm,
                    annot=labels,
                    fmt="",
                    cmap="Blues",
                    mask=mask,
                    xticklabels=["Control", "OTS"],
                    yticklabels=["Control", "OTS"],
                    annot_kws={"size": 12},
                    ax=ax,
                )
                ax.set_ylabel("True labels", fontsize=14)
                ax.set_xlabel("Predicted labels", fontsize=14)
                ax.set_title("Confusion matrix", fontsize=14)
                ax.tick_params(axis="both", labelsize=12)
                cm_plotted = True
                continue

            if plot_ROC and not ROC_plotted:
                # Plot ROC-curve
                display = RocCurveDisplay.from_predictions(
                    y_batches.cpu().numpy(), y_preds.cpu(), plot_chance_level=True, ax=ax
                )
                _ = display.ax_.set_title("ROC curve", fontsize=14)
                display.ax_.set_ylabel("Sensitivity", fontsize=14)
                display.ax_.set_xlabel("1-Specificity", fontsize=14)
                ax.tick_params(axis="both", labelsize=12)
                ROC_plotted = True
                continue

            if plot_PRC and not PRC_plotted:
                # Plot PR-curve
                display = PrecisionRecallDisplay.from_predictions(
                    y_batches.cpu().numpy(), y_preds.cpu(), plot_chance_level=True, ax=ax
                )
                _ = display.ax_.set_title("PR curve", fontsize=14)
                display.ax_.set_ylabel("Precision", fontsize=14)
                display.ax_.set_xlabel("Recall", fontsize=14)
                ax.tick_params(axis="both", labelsize=12)
                PRC_plotted = True
                continue
        fig.suptitle(f"{model_name} (Max F1 ({np.max(th_list):.3f})", fontsize=14)

    # ablation_results[model_name] = np.mean([np.max(th_list), auprc.cpu()])
    if n_plots != 0:
        plt.tight_layout()
        plt.show()

    return auroc, auprc, f1, acc


def test_model_4(
    model, model_name, loader, plot_cm=False, plot_PRC=False, plot_predictions=False, plot_ROC=False
):
    """Evaluate model-4 over all loader batches.

    Returns AUROC, AUPRC, F1, and accuracy; plot flags select diagnostic panels.
    Uses model.threshold, keeps loader order, and disables gradient tracking."""
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        y_preds = torch.empty(0)
        y_batches = torch.empty(0)

        for batch in loader:
            x_batch, y_batch, seq_lengths = batch
            x_batch, y_batch = x_batch.to(device), torch.as_tensor(y_batch, device=device)

            y_pred, _ = model(x_batch)

            y_pred = F.sigmoid(y_pred).flatten()

            y_preds = torch.cat((y_preds, y_pred.cpu()), dim=0)
            y_batches = torch.cat((y_batches, y_batch.cpu()), dim=0)

    auroc = binary_auroc(y_preds, y_batches)
    auprc = binary_auprc(y_preds, y_batches)
    f1 = binary_f1_score(y_preds, y_batches, threshold=model.threshold)
    acc = binary_accuracy(y_preds, y_batches, threshold=model.threshold)

    # Compute avg AUCPR, F1, ACC over varying thresholds
    th = np.linspace(0, 1, 100)
    th_list = []
    for t in th:
        th_list.append(binary_f1_score(y_preds, y_batches, threshold=t).cpu())

    n_plots = sum([plot_cm, plot_PRC, plot_predictions, plot_ROC])

    if n_plots >= 1:
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        axs = axs.flatten()
        pred_plotted, PRC_plotted, cm_plotted, ROC_plotted = False, False, False, False

        for i in range(n_plots):
            ax = axs[i]

            if plot_predictions and not pred_plotted:
                ax.plot(y_preds.cpu(), "o", label="Predicted")
                ax.hlines(
                    model.threshold,
                    0,
                    len(y_preds),
                    "r",
                    linestyles="dashed",
                    label="Decision threshold",
                )
                ax.set_ylim((-0.05, 1.05))
                ax.set_title("Predictions", fontsize=14)
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_ylabel("Probability", fontsize=14)
                ax.tick_params(axis="both", labelsize=12)
                ax.legend()
                pred_plotted = True
                continue

            if plot_cm and not cm_plotted:
                # Plot confusion matrix
                y_pred = [1 if val >= model.threshold else 0 for val in y_preds]
                cm = confusion_matrix(y_batches.cpu(), y_pred)
                TN = cm[0, 0]
                FP = cm[0, 1]
                FN = cm[1, 0]
                TP = cm[1, 1]

                sensitivity = TP / (TP + FN)
                specificity = TN / (TN + FP)
                ppv = TP / (TP + FP)
                npv = TN / (TN + FN)

                cm = np.vstack([cm, np.array([npv, ppv])])
                cm = np.hstack([cm, np.array([specificity, sensitivity, 0]).reshape(3, 1)])

                labels = np.array(
                    [
                        [int(cm[0, 0]), int(cm[0, 1]), f"Specificity\n{cm[0, 2]:.2f}"],
                        [int(cm[1, 0]), int(cm[1, 1]), f"Sensitivity\n{cm[1, 2]:.2f}"],
                        [f"NPV\n{cm[2, 0]:.2f}", f"PPV\n{cm[2, 1]:.2f}", ""],
                    ]
                )

                mask = np.zeros_like(cm, dtype=bool)
                mask[2, 2] = True

                sns.heatmap(
                    cm,
                    annot=labels,
                    fmt="",
                    cmap="Blues",
                    mask=mask,
                    xticklabels=["Control", "OTS"],
                    yticklabels=["Control", "OTS"],
                    annot_kws={"size": 12},
                    ax=ax,
                )
                ax.set_ylabel("True labels", fontsize=14)
                ax.set_xlabel("Predicted labels", fontsize=14)
                ax.set_title("Confusion matrix", fontsize=14)
                ax.tick_params(axis="both", labelsize=12)
                cm_plotted = True
                continue

            if plot_ROC and not ROC_plotted:
                # Plot ROC-curve
                display = RocCurveDisplay.from_predictions(
                    y_batches.cpu().numpy(), y_preds.cpu(), plot_chance_level=True, ax=ax
                )
                _ = display.ax_.set_title("ROC curve", fontsize=14)
                display.ax_.set_ylabel("Sensitivity", fontsize=14)
                display.ax_.set_xlabel("1-Specificity", fontsize=14)
                ax.tick_params(axis="both", labelsize=12)
                ROC_plotted = True
                continue

            if plot_PRC and not PRC_plotted:
                # Plot PR-curve
                display = PrecisionRecallDisplay.from_predictions(
                    y_batches.cpu().numpy(), y_preds.cpu(), plot_chance_level=True, ax=ax
                )
                _ = display.ax_.set_title("PR curve", fontsize=14)
                display.ax_.set_ylabel("Precision", fontsize=14)
                display.ax_.set_xlabel("Recall", fontsize=14)
                ax.tick_params(axis="both", labelsize=12)
                PRC_plotted = True
                continue

        fig.suptitle(f"{model_name} (Max F1 ({np.max(th_list):.3f})", fontsize=14)

    # ablation_results[model_name] = np.mean([np.max(th_list), auprc.cpu()])
    if n_plots != 0:
        plt.tight_layout()
        plt.show()

    return auroc, auprc, f1, acc


def test_model_1_2(
    model,
    model_1,
    model_2,
    model_name,
    loader,
    use_pred=False,
    plot_cm=False,
    plot_PRC=False,
    plot_predictions=False,
    plot_ROC=False,
    return_preds=False,
    save_fig=False,
    output_dir=None,
):
    """Evaluate model-1-2 over all loader batches.

    Returns AUROC, AUPRC, F1, and accuracy; plot flags select diagnostic panels.
    Uses model.threshold, keeps loader order, and disables gradient tracking.
    return_preds appends probabilities and targets. save_fig with plot_ROC saves
    PNG/NPZ files under output_dir (default: configured outputs directory)."""
    device = next(model.parameters()).device
    model.eval()
    model_1.eval()
    model_2.eval()
    prediction_batches, target_batches = [], []
    with torch.no_grad():
        for batch in loader:
            x_batch_clinical, y_batch, x_batch_cpet, _ = batch
            x_batch_clinical, y_batch, x_batch_cpet = (
                x_batch_clinical.float().to(device),
                y_batch.to(device),
                x_batch_cpet.float().to(device),
            )

            # Clinical feature map
            pred_clin, feat_clinical = model_1(x_batch_clinical)

            # CPET feature map
            pred_cpet, feat_cpet = model_2(x_batch_cpet)

            if use_pred:
                y_pred = model(pred_clin, pred_cpet)
            else:
                feature_maps = torch.concatenate((feat_clinical, feat_cpet), axis=1)
                y_pred, _ = model(feature_maps)

            y_pred = F.sigmoid(y_pred).flatten()

            prediction_batches.append(y_pred)
            target_batches.append(y_batch)

    if not prediction_batches:
        raise ValueError("Evaluation loader must contain at least one batch.")
    y_pred = torch.cat(prediction_batches).flatten()
    y_batch = torch.cat(target_batches).flatten()
    auroc = binary_auroc(y_pred, y_batch)
    auprc = binary_auprc(y_pred, y_batch)
    f1 = binary_f1_score(y_pred, y_batch, threshold=model.threshold)
    acc = binary_accuracy(y_pred, y_batch, threshold=model.threshold)

    # Compute avg AUCPR, F1, ACC over varying thresholds
    th = np.linspace(0, 1, 100)
    th_list = []
    for t in th:
        th_list.append(binary_f1_score(y_pred, y_batch, threshold=t).cpu())

    n_plots = sum([plot_cm, plot_PRC, plot_predictions, plot_ROC])

    if n_plots >= 1:
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        axs = axs.flatten()
        pred_plotted, PRC_plotted, cm_plotted, ROC_plotted = False, False, False, False

        for i in range(n_plots):
            ax = axs[i]

            if plot_predictions and not pred_plotted:
                ax.plot(y_pred.cpu(), "o", label="Predicted")
                ax.hlines(
                    model.threshold,
                    0,
                    len(y_pred),
                    "r",
                    linestyles="dashed",
                    label="Decision threshold",
                )
                ax.set_ylim((-0.05, 1.05))
                ax.set_title("Predictions", fontsize=14)
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_ylabel("Probability", fontsize=14)
                ax.tick_params(axis="both", labelsize=12)
                ax.legend()
                pred_plotted = True
                continue

            if plot_cm and not cm_plotted:
                # Plot confusion matrix
                y_preds = [1 if val >= model.threshold else 0 for val in y_pred]
                cm = confusion_matrix(y_batch.cpu(), y_preds)
                TN = cm[0, 0]
                FP = cm[0, 1]
                FN = cm[1, 0]
                TP = cm[1, 1]

                sensitivity = TP / (TP + FN)
                specificity = TN / (TN + FP)
                ppv = TP / (TP + FP)
                npv = TN / (TN + FN)

                cm = np.vstack([cm, np.array([npv, ppv])])
                cm = np.hstack([cm, np.array([specificity, sensitivity, 0]).reshape(3, 1)])

                labels = np.array(
                    [
                        [int(cm[0, 0]), int(cm[0, 1]), f"Specificity\n{cm[0, 2]:.2f}"],
                        [int(cm[1, 0]), int(cm[1, 1]), f"Sensitivity\n{cm[1, 2]:.2f}"],
                        [f"NPV\n{cm[2, 0]:.2f}", f"PPV\n{cm[2, 1]:.2f}", ""],
                    ]
                )

                mask = np.zeros_like(cm, dtype=bool)
                mask[2, 2] = True

                sns.heatmap(
                    cm,
                    annot=labels,
                    fmt="",
                    cmap="Blues",
                    mask=mask,
                    xticklabels=["Control", "OTS"],
                    yticklabels=["Control", "OTS"],
                    annot_kws={"size": 12},
                    ax=ax,
                )
                ax.set_ylabel("True labels", fontsize=14)
                ax.set_xlabel("Predicted labels", fontsize=14)
                ax.set_title("Confusion matrix", fontsize=14)
                ax.tick_params(axis="both", labelsize=12)
                cm_plotted = True
                continue

            if plot_ROC and not ROC_plotted:
                # Plot ROC-curve
                display = RocCurveDisplay.from_predictions(
                    y_batch.cpu().numpy(), y_pred.cpu(), plot_chance_level=True, ax=ax
                )
                _ = display.ax_.set_title("ROC curve", fontsize=14)
                display.ax_.set_ylabel("Sensitivity", fontsize=14)
                display.ax_.set_xlabel("1-Specificity", fontsize=14)
                ax.tick_params(axis="both", labelsize=12)
                ROC_plotted = True

                if save_fig:
                    # Save the ROC curve in its own figure, without the other subplots.
                    roc_fig, roc_ax = plt.subplots(figsize=(4, 4))
                    roc_display = RocCurveDisplay.from_predictions(
                        y_batch.cpu().numpy(),
                        y_pred.cpu(),
                        plot_chance_level=True,
                        ax=roc_ax,
                        color="black",
                    )
                    roc_ax.set_title("ROC curve", fontsize=14)
                    roc_ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=14)
                    roc_ax.set_xlabel("False Positive Rate (1-Specificity)", fontsize=14)
                    roc_ax.tick_params(axis="both", labelsize=12)
                    roc_fig.savefig(
                        output_path("Model_1_2_ROC_curve.png", output_dir),
                        dpi=1200,
                        bbox_inches="tight",
                        facecolor="white",
                    )
                    # Save ROC curve data
                    np.savez(
                        output_path("Model_1_2_ROC_curve.npz", output_dir),
                        fpr=roc_display.fpr,
                        tpr=roc_display.tpr,
                        auc=roc_display.roc_auc,
                    )

                    plt.close(roc_fig)
                continue

            if plot_PRC and not PRC_plotted:
                # Plot PR-curve
                display = PrecisionRecallDisplay.from_predictions(
                    y_batch.cpu().numpy(), y_pred.cpu(), plot_chance_level=True, ax=ax
                )
                _ = display.ax_.set_title("PR curve", fontsize=14)
                display.ax_.set_ylabel("Precision", fontsize=14)
                display.ax_.set_xlabel("Recall", fontsize=14)
                ax.tick_params(axis="both", labelsize=12)
                PRC_plotted = True
                continue
        fig.suptitle(f"{model_name} (Max F1 ({np.max(th_list):.3f})", fontsize=14)

    # ablation_results[model_name] = np.mean([np.max(th_list), auprc.cpu()])
    if n_plots != 0:
        plt.tight_layout()
        plt.show()

    if return_preds:
        return auroc, auprc, f1, acc, y_pred, y_batch
    else:
        return auroc, auprc, f1, acc


def test_model_1_3_4(
    model,
    model_1,
    model_4,
    model_5,
    model_name,
    loader,
    plot_cm=False,
    plot_PRC=False,
    plot_predictions=False,
    plot_ROC=False,
    return_preds=False,
    save_fig=False,
    output_dir=None,
):
    """Evaluate model-1-3-4 over all loader batches.

    Returns AUROC, AUPRC, F1, and accuracy; plot flags select diagnostic panels.
    Uses model.threshold, keeps loader order, and disables gradient tracking.
    return_preds appends probabilities and targets. save_fig with plot_ROC saves
    PNG/NPZ files under output_dir (default: configured outputs directory)."""
    device = next(model.parameters()).device
    model.eval()
    model_1.eval()
    model_4.eval()
    model_5.eval()

    y_preds = torch.empty(0).to(device)
    y_batches = torch.empty(0).to(device)

    with torch.no_grad():
        for batch in loader:
            x_batch_clinical, y_batch, x_batch_bbb, _, x_batch_ecg, _, _ = batch
            x_batch_clinical, y_batch, x_batch_bbb, x_batch_ecg = (
                x_batch_clinical.float().to(device),
                y_batch.to(device),
                x_batch_bbb.float().to(device),
                x_batch_ecg.float().to(device),
            )

            # Clinical & beat-by-beat
            pred_clin, _ = model_1(x_batch_clinical)
            pred_ecg, _ = model_5(x_batch_ecg)

            x_splitted = x_batch_bbb.unfold(dimension=1, size=30, step=15).to(device)
            preds_bbb = torch.empty((0)).to(device)

            for i in range(x_splitted.size(1)):
                win = x_splitted[:, i, :, :]  # (batch, feats, win_len)

                pred_bbb, _, _, _ = model_4(win)

                preds_bbb = torch.cat((preds_bbb, pred_bbb.unsqueeze(0)), dim=0)

            pred_bbb = preds_bbb.mean(0).flatten()

            y_pred = model(
                pred_clin.flatten(), pred_bbb, pred_ecg.flatten()
            )  # Weighted average of predictions

            y_preds = torch.cat((y_preds, y_pred), dim=0)
            # Aggregate ground truths
            y_batches = torch.cat((y_batches, y_batch), dim=0)

    # Compute prediction mean across windows on validation set
    y_preds, y_batches = y_preds.flatten(), y_batches.flatten()

    auroc = binary_auroc(y_preds, y_batches)
    auprc = binary_auprc(y_preds, y_batches)
    f1 = binary_f1_score(y_preds, y_batches, threshold=model.threshold)
    acc = binary_accuracy(y_preds, y_batches, threshold=model.threshold)

    # Compute avg F1, ACC over varying thresholds
    th = np.linspace(0, 1, 100)
    th_list = []
    for t in th:
        th_list.append(binary_f1_score(y_preds, y_batches, threshold=t).cpu())

    n_plots = sum([plot_cm, plot_PRC, plot_predictions, plot_ROC])

    if n_plots >= 1:
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        axs = axs.flatten()
        pred_plotted, PRC_plotted, cm_plotted, ROC_plotted = False, False, False, False

        for i in range(n_plots):
            ax = axs[i]

            if plot_predictions and not pred_plotted:
                ax.plot(y_preds.cpu(), "o", label="Predicted")
                ax.hlines(
                    model.threshold,
                    0,
                    len(y_preds),
                    "r",
                    linestyles="dashed",
                    label="Decision threshold",
                )
                ax.set_ylim((-0.05, 1.05))
                ax.set_title("Predictions", fontsize=14)
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_ylabel("Probability", fontsize=14)
                ax.tick_params(axis="both", labelsize=12)
                ax.legend()
                pred_plotted = True
                continue

            if plot_cm and not cm_plotted:
                # Plot confusion matrix
                y_pred = [1 if val >= model.threshold else 0 for val in y_preds]
                cm = confusion_matrix(y_batches.cpu(), y_pred)
                TN = cm[0, 0]
                FP = cm[0, 1]
                FN = cm[1, 0]
                TP = cm[1, 1]

                sensitivity = TP / (TP + FN)
                specificity = TN / (TN + FP)
                ppv = TP / (TP + FP)
                npv = TN / (TN + FN)

                cm = np.vstack([cm, np.array([npv, ppv])])
                cm = np.hstack([cm, np.array([specificity, sensitivity, 0]).reshape(3, 1)])

                labels = np.array(
                    [
                        [int(cm[0, 0]), int(cm[0, 1]), f"Specificity\n{cm[0, 2]:.2f}"],
                        [int(cm[1, 0]), int(cm[1, 1]), f"Sensitivity\n{cm[1, 2]:.2f}"],
                        [f"NPV\n{cm[2, 0]:.2f}", f"PPV\n{cm[2, 1]:.2f}", ""],
                    ]
                )

                mask = np.zeros_like(cm, dtype=bool)
                mask[2, 2] = True

                sns.heatmap(
                    cm,
                    annot=labels,
                    fmt="",
                    cmap="Blues",
                    mask=mask,
                    xticklabels=["Control", "OTS"],
                    yticklabels=["Control", "OTS"],
                    annot_kws={"size": 12},
                    ax=ax,
                )
                ax.set_ylabel("True labels", fontsize=14)
                ax.set_xlabel("Predicted labels", fontsize=14)
                ax.set_title("Confusion matrix", fontsize=14)
                ax.tick_params(axis="both", labelsize=12)
                cm_plotted = True
                continue

            if plot_ROC and not ROC_plotted:
                # Plot ROC-curve
                display = RocCurveDisplay.from_predictions(
                    y_batches.cpu().numpy(), y_preds.cpu(), plot_chance_level=True, ax=ax
                )
                _ = display.ax_.set_title("ROC curve", fontsize=14)
                display.ax_.set_ylabel("Sensitivity", fontsize=14)
                display.ax_.set_xlabel("1-Specificity", fontsize=14)
                ax.tick_params(axis="both", labelsize=12)
                ROC_plotted = True

                if save_fig:
                    # Save the ROC curve in its own figure, without the other subplots.
                    roc_fig, roc_ax = plt.subplots(figsize=(4, 4))
                    roc_display = RocCurveDisplay.from_predictions(
                        y_batches.cpu().numpy(),
                        y_preds.cpu(),
                        plot_chance_level=True,
                        ax=roc_ax,
                        color="black",
                    )
                    roc_ax.set_title("ROC curve", fontsize=14)
                    roc_ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=14)
                    roc_ax.set_xlabel("False Positive Rate (1-Specificity)", fontsize=14)
                    roc_ax.tick_params(axis="both", labelsize=12)
                    roc_fig.savefig(
                        output_path("Model_1_3_4_ROC_curve.png", output_dir),
                        dpi=1200,
                        bbox_inches="tight",
                        facecolor="white",
                    )
                    # Save ROC curve data
                    np.savez(
                        output_path("Model_1_3_4_ROC_curve.npz", output_dir),
                        fpr=roc_display.fpr,
                        tpr=roc_display.tpr,
                        auc=roc_display.roc_auc,
                    )

                    plt.close(roc_fig)
                continue

            if plot_PRC and not PRC_plotted:
                # Plot PR-curve
                display = PrecisionRecallDisplay.from_predictions(
                    y_batches.cpu().numpy(), y_preds.cpu(), plot_chance_level=True, ax=ax
                )
                _ = display.ax_.set_title("PR curve", fontsize=14)
                display.ax_.set_ylabel("Precision", fontsize=14)
                display.ax_.set_xlabel("Recall", fontsize=14)
                ax.tick_params(axis="both", labelsize=12)
                PRC_plotted = True
                continue
        fig.suptitle(f"{model_name} (Max F1 ({np.max(th_list):.3f})", fontsize=14)

    # ablation_results[model_name] = np.mean([np.max(th_list), auprc.cpu()])
    if n_plots != 0:
        plt.tight_layout()
        plt.show()

    if return_preds:
        return auroc, auprc, f1, acc, y_preds, y_batches
    else:
        return auroc, auprc, f1, acc
