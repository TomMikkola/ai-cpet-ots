"""Original five-fold Bayesian hyperparameter searches for the six models.

Install the optimization extra before importing this module. Objectives,
search grids, early stopping, and unseeded shuffled folds follow the study.
Only threshold selection now has a deterministic all-zero-score fallback."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from GPyOpt.methods import BayesianOptimization
from sklearn.metrics import average_precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics.functional import binary_accuracy, binary_auprc, binary_f1_score

from ai_cpet_ots.dataloaders import (
    Model_1_2_dataset,
    Model_1_3_4_dataset,
    Model_3_Dataset,
    Model_4_Dataset,
)
from ai_cpet_ots.helpers import collate, collate_model_1_3_4
from ai_cpet_ots.models import Model_1, Model_1_2, Model_1_3_4, Model_2, Model_3, Model_4
from ai_cpet_ots.paths import MODEL_WEIGHTS


class Optimizer_Model_1:
    """Bayesian search for Model-1.

    Construct with the training arrays only; validation folds are created by the
    objective. Training data is stored on the instance, not loaded from disk."""

    def __init__(self, X_train, Y_train):
        """Initialize Optimizer_Model_1 with the supplied study configuration."""
        self.threshold = None
        self.best_score = 999
        self.X_train = X_train
        self.Y_train = Y_train

    def optimize_threshold(self, y_pred, y_true, metric):
        """Search the original 50-point [0, 1] grid for AP, F1, or recall.

        Returns the first improving threshold, or 0.0 when every score is zero.
        Metric names are case-sensitive; unsupported names raise ValueError."""
        threshold = 0.0  # First grid point is the deterministic fallback for all-zero scores.
        if metric not in {"AP", "F1", "recall"}:
            raise ValueError(f"Unsupported threshold metric: {metric}")
        if metric == "AP":
            ap_temp = 0
            for th in np.linspace(0, 1):
                y_pred_th = [1 if val > th else 0 for val in y_pred]
                temp = average_precision_score(y_true.cpu(), y_pred_th)
                if temp > ap_temp:
                    ap_temp = temp
                    threshold = th
        elif metric == "F1":
            f1_temp = 0
            for th in np.linspace(0, 1):
                temp = binary_f1_score(y_pred, y_true, threshold=th)
                if temp > f1_temp:
                    f1_temp = temp
                    threshold = th
        elif metric == "recall":
            recall_temp = 0
            for th in np.linspace(0, 1):
                y_pred_th = [1 if val > th else 0 for val in y_pred]
                temp = recall_score(y_true.cpu(), y_pred_th)
                if temp > recall_temp:
                    recall_temp = temp
                    threshold = th

        return threshold

    def objective_func(self, params):
        """Train the model across five shuffled stratified folds for one parameter row.

        params is the two-dimensional array supplied by GPyOpt. Returns the original
        scalar study objective and updates the best threshold and mean epoch count.
        Fold shuffling remains unseeded to preserve the existing research behavior."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        params = params[0]
        n_epochs, dropout, lr, weight_decay, batch_size, n_hidden_layers, n_nodes = params
        n_epochs, batch_size, n_hidden_layers, n_nodes = (
            int(n_epochs),
            int(batch_size),
            int(n_hidden_layers),
            int(n_nodes),
        )

        print(
            f"***PARAMS n_epochs: {n_epochs}, dropout: {dropout}, lr: {lr}, weight_decay: {weight_decay}, batch_size: {batch_size}, n_hidden_layers {n_hidden_layers}, n_nodes {n_nodes}***"
        )

        auprc_list = []
        f1_list = []
        acc_list = []
        val_loss_list = []
        threshold_list = []
        self.n_epochs = []
        scores = []

        skf = StratifiedKFold(n_splits=5, shuffle=True)

        for k, (train_index, val_index) in enumerate(skf.split(self.X_train, self.Y_train)):
            print(f"Kfold {k}")
            X_train = [self.X_train[i] for i in train_index]
            Y_train = [self.Y_train[i] for i in train_index]
            x_val = [self.X_train[i] for i in val_index]
            y_val = [self.Y_train[i] for i in val_index]

            trainset = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
            trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
            valset = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))
            valloader = DataLoader(valset, batch_size=len(valset))

            model = Model_1(dropout, n_hidden_layers, n_nodes).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            # Early stopping parameters
            last_val_loss = 999
            es_count = 0
            patience = 3
            delta = 1e-4
            stop_training = False

            pos_samples = np.sum(Y_train)
            neg_samples = len(Y_train) - pos_samples
            pos_weight = torch.tensor([neg_samples / pos_samples]).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            for e in range(n_epochs):
                # Training
                model.train()
                train_loss = 0
                val_loss = 0

                for batch in trainloader:
                    x_batch, y_batch = batch
                    x_batch, y_batch = x_batch.float().to(device), y_batch.to(device)

                    y_pred, _ = model(x_batch)

                    loss = criterion(y_pred.flatten(), y_batch)
                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    train_loss += loss.item() / len(trainloader)

                # Evaluation
                model.eval()
                y_preds = torch.empty(0).to(device)
                y_batches = torch.empty(0).to(device)

                if torch.isnan(loss):
                    train_loss = 999

                with torch.no_grad():
                    for batch in valloader:
                        x_batch, y_batch = batch
                        x_batch, y_batch = x_batch.float().to(device), y_batch.to(device)

                        y_pred, _ = model(x_batch)

                        v_loss = criterion(y_pred.flatten(), y_batch.float())
                        val_loss += v_loss.item() / len(valloader)

                        y_pred = F.sigmoid(y_pred).flatten()
                        y_preds = torch.cat((y_preds, y_pred), dim=0)
                        y_batches = torch.cat((y_batches, y_batch), dim=0)

                if torch.isnan(v_loss):
                    val_loss = 999

                print(f"Epoch {e + 1}/{n_epochs} - train loss {train_loss} - val loss {val_loss}")

                # Early stopping
                if (last_val_loss - val_loss) < delta:
                    es_count += 1
                    if es_count >= patience:
                        stop_training = True
                else:
                    es_count = 0

                if stop_training:
                    print(f"Early stop at {e + 1}")
                    break

                last_val_loss = val_loss

            self.n_epochs.append(e + 1)
            threshold = self.optimize_threshold(y_preds, y_batches, "F1")
            threshold_list.append(threshold)

            auprc = binary_auprc(y_preds, y_batches)
            f1 = binary_f1_score(y_preds, y_batches, threshold=threshold)
            acc = binary_accuracy(y_preds, y_batches, threshold=threshold)
            auprc_list.append(auprc.item())
            f1_list.append(f1.item())
            acc_list.append(acc.item())
            val_loss_list.append(last_val_loss)
            scores.append(last_val_loss - f1.item())

        score = np.mean(scores)

        print(
            f"AUCPR: {np.mean(auprc_list):.3f} ({np.std(auprc_list):.3f}), F1: {np.mean(f1_list):.3f} ({np.std(f1_list):.3f}), ACC: {np.mean(acc_list):.3f} ({np.std(acc_list):.3f}), val_loss {np.mean(val_loss_list):.4f} ({np.std(val_loss_list):.4f}), score {score:.4f} ({np.std(scores):.4f}), threshold {np.mean(threshold_list):.3f} ({np.std(threshold_list):.3f})"
        )

        if score < self.best_score:
            self.best_score = score
            self.threshold = np.mean(threshold_list)

        self.n_epochs = int(np.ceil(np.mean(self.n_epochs)))

        return score

    def run_bo_gp(self, bounds, max_iter):
        """Run the original Gaussian-process Bayesian search over GPyOpt bounds.

        max_iter controls search iterations. Returns [best_objective, parameters]
        with the selected threshold appended to parameters; records self.n_epochs."""
        self.threshold = None
        self.best_score = 999

        bo_gp = BayesianOptimization(
            f=self.objective_func, domain=bounds, acquisition_optimizer_type="CMA"
        )

        bo_gp.run_optimization(max_iter=max_iter)
        print(f"Best parameters found with score {bo_gp.fx_opt}:")
        print(bo_gp.x_opt)

        return [bo_gp.fx_opt, np.append(bo_gp.x_opt, self.threshold)]


class Optimizer_Model_2:
    """Bayesian search for Model-2.

    Construct with the training arrays only; validation folds are created by the
    objective. Training data is stored on the instance, not loaded from disk."""

    def __init__(self, X_train, Y_train):
        """Initialize Optimizer_Model_2 with the supplied study configuration."""
        self.threshold = None
        self.best_score = 999
        self.X_train = X_train
        self.Y_train = Y_train

    def optimize_threshold(self, y_pred, y_true, metric):
        """Search the original 50-point [0, 1] grid for AP, F1, or recall.

        Returns the first improving threshold, or 0.0 when every score is zero.
        Metric names are case-sensitive; unsupported names raise ValueError."""
        threshold = 0.0  # First grid point is the deterministic fallback for all-zero scores.
        if metric not in {"AP", "F1", "recall"}:
            raise ValueError(f"Unsupported threshold metric: {metric}")
        if metric == "AP":
            ap_temp = 0
            for th in np.linspace(0, 1):
                y_pred_th = [1 if val > th else 0 for val in y_pred]
                temp = average_precision_score(y_true.cpu(), y_pred_th)
                if temp > ap_temp:
                    ap_temp = temp
                    threshold = th
        elif metric == "F1":
            f1_temp = 0
            for th in np.linspace(0, 1):
                temp = binary_f1_score(y_pred, y_true, threshold=th)
                if temp > f1_temp:
                    f1_temp = temp
                    threshold = th
        elif metric == "recall":
            recall_temp = 0
            for th in np.linspace(0, 1):
                y_pred_th = [1 if val > th else 0 for val in y_pred]
                temp = recall_score(y_true.cpu(), y_pred_th)
                if temp > recall_temp:
                    recall_temp = temp
                    threshold = th

        return threshold

    def objective_func(self, params):
        """Train the model across five shuffled stratified folds for one parameter row.

        params is the two-dimensional array supplied by GPyOpt. Returns the original
        scalar study objective and updates the best threshold and mean epoch count.
        Fold shuffling remains unseeded to preserve the existing research behavior."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        params = params[0]
        n_epochs, dropout, lr, weight_decay, batch_size, n_hidden_layers, n_nodes = params
        n_epochs, batch_size, n_hidden_layers, n_nodes = (
            int(n_epochs),
            int(batch_size),
            int(n_hidden_layers),
            int(n_nodes),
        )

        print(
            f"***PARAMS n_epochs: {n_epochs}, dropout: {dropout}, lr: {lr}, weight_decay: {weight_decay}, batch_size: {batch_size}, n_hidden_layers {n_hidden_layers}, n_nodes {n_nodes}***"
        )

        auprc_list = []
        f1_list = []
        acc_list = []
        val_loss_list = []
        threshold_list = []
        self.n_epochs = []
        scores = []

        skf = StratifiedKFold(n_splits=5, shuffle=True)

        for k, (train_index, val_index) in enumerate(skf.split(self.X_train, self.Y_train)):
            print(f"Kfold {k}")
            x_train = [self.X_train[i] for i in train_index]
            y_train = [self.Y_train[i] for i in train_index]
            x_val = [self.X_train[i] for i in val_index]
            y_val = [self.Y_train[i] for i in val_index]

            trainset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
            trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
            valset = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))
            valloader = DataLoader(valset, batch_size=len(valset))

            model = Model_2(dropout, n_hidden_layers, n_nodes).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            # Early stopping parameters
            last_val_loss = 999
            es_count = 0
            patience = 3
            delta = 1e-4
            stop_training = False

            pos_samples = np.sum(y_train)
            neg_samples = len(y_train) - pos_samples
            pos_weight = torch.tensor([neg_samples / pos_samples]).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            for e in range(n_epochs):
                # Training
                model.train()
                train_loss = 0
                val_loss = 0

                for batch in trainloader:
                    x_batch, y_batch = batch
                    x_batch, y_batch = x_batch.float().to(device), y_batch.to(device)

                    y_pred, _ = model(x_batch)

                    loss = criterion(y_pred.flatten(), y_batch)
                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    train_loss += loss.item() / len(trainloader)

                # Evaluation
                model.eval()
                y_preds = torch.empty(0).to(device)
                y_batches = torch.empty(0).to(device)

                if torch.isnan(loss):
                    train_loss = 999

                with torch.no_grad():
                    for batch in valloader:
                        x_batch, y_batch = batch
                        x_batch, y_batch = x_batch.float().to(device), y_batch.to(device)

                        y_pred, _ = model(x_batch)

                        v_loss = criterion(y_pred.flatten(), y_batch.float())
                        val_loss += v_loss.item() / len(valloader)

                        y_pred = F.sigmoid(y_pred).flatten()
                        y_preds = torch.cat((y_preds, y_pred), dim=0)
                        y_batches = torch.cat((y_batches, y_batch), dim=0)

                if torch.isnan(v_loss):
                    val_loss = 999

                print(f"Epoch {e + 1}/{n_epochs} - train loss {train_loss} - val loss {val_loss}")

                # Early stopping
                if (last_val_loss - val_loss) < delta:
                    es_count += 1
                    if es_count >= patience:
                        stop_training = True
                else:
                    es_count = 0

                if stop_training:
                    print(f"Early stop at {e + 1}")
                    break

                last_val_loss = val_loss

            self.n_epochs.append(e + 1)
            threshold = self.optimize_threshold(y_preds, y_batches, "F1")
            threshold_list.append(threshold)

            auprc = binary_auprc(y_preds, y_batches)
            f1 = binary_f1_score(y_preds, y_batches, threshold=threshold)
            acc = binary_accuracy(y_preds, y_batches, threshold=threshold)
            auprc_list.append(auprc.item())
            f1_list.append(f1.item())
            acc_list.append(acc.item())
            val_loss_list.append(last_val_loss)
            scores.append(last_val_loss - f1.item())

        score = np.mean(scores)

        print(
            f"AUCPR: {np.mean(auprc_list):.3f} ({np.std(auprc_list):.3f}), F1: {np.mean(f1_list):.3f} ({np.std(f1_list):.3f}), ACC: {np.mean(acc_list):.3f} ({np.std(acc_list):.3f}), val_loss {np.mean(val_loss_list):.4f} ({np.std(val_loss_list):.4f}), score {score:.4f} ({np.std(scores):.4f}), threshold {np.mean(threshold_list):.3f} ({np.std(threshold_list):.3f})"
        )

        if score < self.best_score:
            self.best_score = score
            self.threshold = np.mean(threshold_list)

        self.n_epochs = int(np.ceil(np.mean(self.n_epochs)))

        return score

    def run_bo_gp(self, bounds, max_iter):
        """Run the original Gaussian-process Bayesian search over GPyOpt bounds.

        max_iter controls search iterations. Returns [best_objective, parameters]
        with the selected threshold appended to parameters; records self.n_epochs."""
        self.threshold = None
        self.best_score = 999

        bo_gp = BayesianOptimization(
            f=self.objective_func, domain=bounds, acquisition_optimizer_type="CMA"
        )

        bo_gp.run_optimization(max_iter=max_iter)
        print(f"Best parameters found with score {bo_gp.fx_opt}:")
        print(bo_gp.x_opt)

        return [bo_gp.fx_opt, np.append(bo_gp.x_opt, self.threshold)]


class Optimizer_Model_3:
    """Bayesian search for Model-3.

    Construct with the training arrays only; validation folds are created by the
    objective. Training data is stored on the instance, not loaded from disk."""

    def __init__(self, X_train, Y_train):
        """Initialize Optimizer_Model_3 with the supplied study configuration."""
        self.threshold = None
        self.best_score = 999
        self.X_train = X_train
        self.Y_train = Y_train

    def optimize_threshold(self, y_pred, y_true, metric):
        """Search the original 50-point [0, 1] grid for AP, F1, or recall.

        Returns the first improving threshold, or 0.0 when every score is zero.
        Metric names are case-sensitive; unsupported names raise ValueError."""
        threshold = 0.0  # First grid point is the deterministic fallback for all-zero scores.
        if metric not in {"AP", "F1", "recall"}:
            raise ValueError(f"Unsupported threshold metric: {metric}")
        if metric == "AP":
            ap_temp = 0
            for th in np.linspace(0, 1):
                y_pred_th = [1 if val > th else 0 for val in y_pred]
                temp = average_precision_score(y_true.cpu(), y_pred_th)
                if temp > ap_temp:
                    ap_temp = temp
                    threshold = th
        elif metric == "F1":
            f1_temp = 0
            for th in np.linspace(0, 1):
                temp = binary_f1_score(y_pred, y_true, threshold=th)
                if temp > f1_temp:
                    f1_temp = temp
                    threshold = th
        elif metric == "recall":
            recall_temp = 0
            for th in np.linspace(0, 1):
                y_pred_th = [1 if val > th else 0 for val in y_pred]
                temp = recall_score(y_true.cpu(), y_pred_th)
                if temp > recall_temp:
                    recall_temp = temp
                    threshold = th

        return threshold

    def objective_func(self, params):
        """Train the model across five shuffled stratified folds for one parameter row.

        params is the two-dimensional array supplied by GPyOpt. Returns the original
        scalar study objective and updates the best threshold and mean epoch count.
        Fold shuffling remains unseeded to preserve the existing research behavior."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        params = params[0]
        n_epochs, dropout, lr, weight_decay, batch_size = params
        n_epochs, batch_size = int(n_epochs), int(batch_size)

        print(
            f"***PARAMS n_epochs: {n_epochs}, dropout: {dropout}, lr: {lr}, weight_decay: {weight_decay}, batch_size: {batch_size}***"
        )

        auprc_list = []
        f1_list = []
        acc_list = []
        val_loss_list = []
        threshold_list = []
        self.n_epochs = []
        scores = []

        skf = StratifiedKFold(n_splits=5, shuffle=True)

        for k, (train_index, val_index) in enumerate(skf.split(self.X_train, self.Y_train)):
            print(f"Kfold {k}")
            x_train = [self.X_train[i] for i in train_index]
            y_train = [self.Y_train[i] for i in train_index]
            x_val = [self.X_train[i] for i in val_index]
            y_val = [self.Y_train[i] for i in val_index]

            # Shuffle trainset and create minibatch
            trainset = Model_3_Dataset(x_train, y_train)
            trainloader = DataLoader(
                trainset, batch_size=batch_size, shuffle=True, collate_fn=collate, pin_memory=True
            )
            valset = Model_3_Dataset(x_val, y_val)
            valloader = DataLoader(
                valset, batch_size=len(valset), collate_fn=collate, pin_memory=True
            )

            model = Model_3(dropout).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            # Early stopping parameters
            last_val_loss = 999
            es_count = 0
            patience = 3
            delta = 1e-4
            stop_training = False

            pos_samples = np.sum(y_train)
            neg_samples = len(y_train) - pos_samples
            pos_weight = torch.tensor([neg_samples / pos_samples]).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            for e in range(n_epochs):
                # Training
                model.train()
                train_loss = 0
                val_loss = 0

                for batch in trainloader:
                    x_batch, y_batch, _ = batch
                    x_batch, y_batch = x_batch.float().to(device), torch.tensor(y_batch).to(device)

                    y_pred, _, _, _ = model(x_batch)
                    loss = criterion(y_pred.flatten(), y_batch.float())
                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    train_loss += loss.item() / len(trainloader)

                # Evaluation
                model.eval()
                y_preds = torch.empty(0).to(device)
                y_batches = torch.empty(0).to(device)

                if torch.isnan(loss):
                    train_loss = 999

                with torch.no_grad():
                    for batch in valloader:
                        x_batch, y_batch, _ = batch
                        x_batch, y_batch = (
                            x_batch.float().to(device),
                            torch.tensor(y_batch).to(device),
                        )

                        y_pred, _, _, _ = model(x_batch)
                        v_loss = criterion(y_pred.flatten(), y_batch.float())
                        val_loss += v_loss.item() / len(valloader)

                        y_pred = F.sigmoid(y_pred).flatten()
                        y_preds = torch.cat((y_preds, y_pred), dim=0)
                        y_batches = torch.cat((y_batches, y_batch), dim=0)

                if torch.isnan(v_loss):
                    val_loss = 999

                print(f"Epoch {e + 1}/{n_epochs} - train loss {train_loss} - val loss {val_loss}")

                # Early stopping
                if (last_val_loss - val_loss) < delta:
                    es_count += 1
                    if es_count >= patience:
                        stop_training = True
                else:
                    es_count = 0

                if stop_training:
                    print(f"Early stop at {e + 1}")
                    break

                last_val_loss = val_loss

            self.n_epochs.append(e + 1)
            threshold = self.optimize_threshold(y_preds, y_batches, "F1")
            threshold_list.append(threshold)

            auprc = binary_auprc(y_preds, y_batches)
            f1 = binary_f1_score(y_preds, y_batches, threshold=threshold)
            acc = binary_accuracy(y_preds, y_batches, threshold=threshold)
            auprc_list.append(auprc.item())
            f1_list.append(f1.item())
            acc_list.append(acc.item())
            val_loss_list.append(last_val_loss)
            scores.append(last_val_loss - f1.item())

        score = np.mean(scores)

        print(
            f"AUCPR: {np.mean(auprc_list):.3f} ({np.std(auprc_list):.3f}), F1: {np.mean(f1_list):.3f} ({np.std(f1_list):.3f}), ACC: {np.mean(acc_list):.3f} ({np.std(acc_list):.3f}), val_loss {np.mean(val_loss_list):.4f} ({np.std(val_loss_list):.4f}), score {score:.4f} ({np.std(scores):.4f}), threshold {np.mean(threshold_list):.3f} ({np.std(threshold_list):.3f})"
        )

        if score < self.best_score:
            self.best_score = score
            self.threshold = np.mean(threshold_list)

        self.n_epochs = int(np.ceil(np.mean(self.n_epochs)))

        return score

    def run_bo_gp(self, bounds, max_iter, X_init=None, Y_init=None):
        """Run the original Gaussian-process Bayesian search over GPyOpt bounds.

        max_iter controls search iterations. Returns [best_objective, parameters]
        with the selected threshold appended to parameters; records self.n_epochs."""
        self.threshold = None
        self.best_score = 999

        bo_gp = BayesianOptimization(
            f=self.objective_func,
            domain=bounds,
            acquisition_optimizer_type="CMA",
            X=X_init,
            Y=Y_init,
        )

        bo_gp.run_optimization(max_iter=max_iter)
        print(f"Best parameters found with score {bo_gp.fx_opt}:")
        print(bo_gp.x_opt)

        return [bo_gp.fx_opt, np.append(bo_gp.x_opt, self.threshold)]


class Optimizer_Model_4:
    """Bayesian search for Model-4.

    Construct with the training arrays only; validation folds are created by the
    objective. Training data is stored on the instance, not loaded from disk."""

    def __init__(self, X_train, Y_train):
        """Initialize Optimizer_Model_4 with the supplied study configuration."""
        self.threshold = None
        self.best_score = 999
        self.X_train = X_train
        self.Y_train = Y_train

    def optimize_threshold(self, y_pred, y_true, metric):
        """Search the original 50-point [0, 1] grid for AP, F1, or recall.

        Returns the first improving threshold, or 0.0 when every score is zero.
        Metric names are case-sensitive; unsupported names raise ValueError."""
        threshold = 0.0  # First grid point is the deterministic fallback for all-zero scores.
        if metric not in {"AP", "F1", "recall"}:
            raise ValueError(f"Unsupported threshold metric: {metric}")
        if metric == "AP":
            ap_temp = 0
            for th in np.linspace(0, 1):
                y_pred_th = [1 if val > th else 0 for val in y_pred]
                temp = average_precision_score(y_true.cpu(), y_pred_th)
                if temp > ap_temp:
                    ap_temp = temp
                    threshold = th
        elif metric == "F1":
            f1_temp = 0
            for th in np.linspace(0, 1):
                temp = binary_f1_score(y_pred, y_true, threshold=th)
                if temp > f1_temp:
                    f1_temp = temp
                    threshold = th
        elif metric == "recall":
            recall_temp = 0
            for th in np.linspace(0, 1):
                y_pred_th = [1 if val > th else 0 for val in y_pred]
                temp = recall_score(y_true.cpu(), y_pred_th)
                if temp > recall_temp:
                    recall_temp = temp
                    threshold = th

        return threshold

    def objective_func(self, params):
        """Train the model across five shuffled stratified folds for one parameter row.

        params is the two-dimensional array supplied by GPyOpt. Returns the original
        scalar study objective and updates the best threshold and mean epoch count.
        Fold shuffling remains unseeded to preserve the existing research behavior."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        try:
            n_epochs, dropout, lr, weight_decay = params[0]
            n_epochs = int(n_epochs)
        except (ValueError, TypeError):
            n_epochs, dropout, lr, weight_decay, batch_size = params[0]
            n_epochs, batch_size = int(n_epochs), int(batch_size)

        auprc_list = []
        f1_list = []
        acc_list = []
        val_loss_list = []
        threshold_list = []
        self.n_epochs = []
        scores = []

        try:
            print(
                f"***PARAMS n_epochs: {n_epochs}, dropout: {dropout}, lr: {lr}, weight_decay: {weight_decay}, batch_size: {batch_size}***"
            )
        except UnboundLocalError:
            print(
                f"***PARAMS n_epochs: {n_epochs}, lr: {lr}, weight_decay: {weight_decay}, batch_size: {batch_size}***"
            )

        skf = StratifiedKFold(n_splits=5, shuffle=True)

        for k, (train_index, val_index) in enumerate(skf.split(self.X_train, self.Y_train)):
            print(f"Kfold {k}")
            x_train = [self.X_train[i] for i in train_index]
            y_train = [self.Y_train[i] for i in train_index]
            x_val = [self.X_train[i] for i in val_index]
            y_val = [self.Y_train[i] for i in val_index]

            # Shuffle trainset and create minibatch
            trainset = Model_4_Dataset(x_train, y_train)
            trainloader = DataLoader(
                trainset, batch_size=batch_size, shuffle=True, collate_fn=collate
            )
            valset = Model_4_Dataset(x_val, y_val)
            valloader = DataLoader(valset, batch_size=batch_size, collate_fn=collate)

            model = Model_4(dropout).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            # Early stopping parameters
            last_val_loss = 0
            es_count = 0
            patience = 5
            delta = 1e-4
            stop_training = False

            pos_samples = np.sum(y_train)
            neg_samples = len(y_train) - pos_samples
            pos_weight = torch.tensor([neg_samples / pos_samples]).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            for e in range(n_epochs):
                # Training
                model.train()
                train_loss = 0
                val_loss = 0
                for batch in trainloader:
                    x_batch, y_batch, _ = batch
                    x_batch, y_batch = x_batch.float().to(device), torch.tensor(y_batch).to(device)

                    y_pred, _ = model(x_batch)

                    loss = criterion(y_pred.flatten(), y_batch.float())
                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    train_loss += loss.item() / len(trainloader)

                if torch.isnan(loss):
                    train_loss = 999

                # Evaluation
                model.eval()
                y_preds = torch.empty((0)).to(device)
                y_batches = torch.empty((0)).to(device)

                with torch.no_grad():
                    for batch in valloader:
                        x_batch, y_batch, seq_lengths = batch
                        x_batch, y_batch = (
                            x_batch.float().to(device),
                            torch.tensor(y_batch).to(device),
                        )

                        y_pred, _ = model(x_batch)

                        v_loss = criterion(y_pred.flatten(), y_batch.float())
                        val_loss += v_loss.item() / len(valloader)

                        y_pred = F.sigmoid(y_pred).flatten()
                        y_preds = torch.cat((y_preds, y_pred), dim=0)
                        y_batches = torch.cat((y_batches, y_batch), dim=0)

                if torch.isnan(v_loss):
                    val_loss = 999

                print(f"Epoch {e + 1}/{n_epochs} - train loss {train_loss} - val loss {val_loss}")

                # Early stopping
                if (last_val_loss - val_loss) < delta:
                    es_count += 1
                    if es_count >= patience:
                        stop_training = True
                else:
                    es_count = 0

                if stop_training:
                    print(f"Early stop at {e + 1}")
                    break

                last_val_loss = val_loss

            self.n_epochs.append(e + 1)
            threshold = self.optimize_threshold(y_preds, y_batches, "F1")
            threshold_list.append(threshold)

            auprc = binary_auprc(y_preds, y_batches)
            f1 = binary_f1_score(y_preds, y_batches, threshold=threshold)
            acc = binary_accuracy(y_preds, y_batches, threshold=threshold)
            auprc_list.append(auprc.item())
            f1_list.append(f1.item())
            acc_list.append(acc.item())
            val_loss_list.append(last_val_loss)
            scores.append(last_val_loss - f1.item())

        score = np.mean(scores)

        print(
            f"AUCPR: {np.mean(auprc_list):.3f} ({np.std(auprc_list):.3f}), F1: {np.mean(f1_list):.3f} ({np.std(f1_list):.3f}), ACC: {np.mean(acc_list):.3f} ({np.std(acc_list):.3f}), val_loss {np.mean(val_loss_list):.4f} ({np.std(val_loss_list):.4f}), score {score:.4f} ({np.std(scores):.4f}), threshold {np.mean(threshold_list):.3f} ({np.std(threshold_list):.3f})"
        )

        if score < self.best_score:
            self.best_score = score
            self.threshold = np.mean(threshold_list)

        self.n_epochs = int(np.ceil(np.mean(self.n_epochs)))

        return score

    def run_bo_gp(self, bounds, max_iter, X_init=None, Y_init=None):
        """Run the original Gaussian-process Bayesian search over GPyOpt bounds.

        max_iter controls search iterations. Returns [best_objective, parameters]
        with the selected threshold appended to parameters; records self.n_epochs."""
        self.threshold = None
        self.best_score = 999

        bo_gp = BayesianOptimization(
            f=self.objective_func,
            domain=bounds,
            acquisition_optimizer_type="CMA",  # DIRECT,
            X=X_init,
            Y=Y_init,
            initial_design_numdata=3,
        )

        bo_gp.run_optimization(max_iter=max_iter)
        print(f"Best parameters found with score {bo_gp.fx_opt}:")
        print(bo_gp.x_opt)

        return [bo_gp.fx_opt, np.append(bo_gp.x_opt, self.threshold)]


class Optimizer_Model_1_2:
    """Bayesian search for Model-1-2.

    Construct with the training arrays only; validation folds are created by the
    objective. Training data is stored on the instance, not loaded from disk."""

    def __init__(
        self, X_train_clinical, Y_train_clinical, X_train_cpet, Y_train_cpet, use_pred=False
    ):
        """Initialize Optimizer_Model_1_2 with the supplied study configuration."""
        self.threshold = None
        self.best_score = 999
        self.use_pred = use_pred
        self.X_train_clinical = X_train_clinical
        self.Y_train_clinical = Y_train_clinical
        self.X_train_cpet = X_train_cpet
        self.Y_train_cpet = Y_train_cpet

    def optimize_threshold(self, y_pred, y_true, metric):
        """Search the original 50-point [0, 1] grid for AP, F1, or recall.

        Returns the first improving threshold, or 0.0 when every score is zero.
        Metric names are case-sensitive; unsupported names raise ValueError."""
        threshold = 0.0  # First grid point is the deterministic fallback for all-zero scores.
        if metric not in {"AP", "F1", "recall"}:
            raise ValueError(f"Unsupported threshold metric: {metric}")
        if metric == "AP":
            ap_temp = 0
            for th in np.linspace(0, 1):
                y_pred_th = [1 if val > th else 0 for val in y_pred]
                temp = average_precision_score(y_true.cpu(), y_pred_th)
                if temp > ap_temp:
                    ap_temp = temp
                    threshold = th
        elif metric == "F1":
            f1_temp = 0
            for th in np.linspace(0, 1):
                temp = binary_f1_score(y_pred, y_true, threshold=th)
                if temp > f1_temp:
                    f1_temp = temp
                    threshold = th
        elif metric == "recall":
            recall_temp = 0
            for th in np.linspace(0, 1):
                y_pred_th = [1 if val > th else 0 for val in y_pred]
                temp = recall_score(y_true.cpu(), y_pred_th)
                if temp > recall_temp:
                    recall_temp = temp
                    threshold = th

        return threshold

    def objective_func(self, params):
        """Train the model across five shuffled stratified folds for one parameter row.

        params is the two-dimensional array supplied by GPyOpt. Returns the original
        scalar study objective and updates the best threshold and mean epoch count.
        Fold shuffling remains unseeded to preserve the existing research behavior."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        params = params[0]
        if self.use_pred:
            n_epochs, lr, weight_decay, batch_size = params
            n_epochs, batch_size = int(n_epochs), int(batch_size)

            print(
                f"***PARAMS n_epochs: {n_epochs}, lr: {lr}, weight_decay: {weight_decay}, batch_size: {batch_size}***"
            )
        else:
            n_epochs, dropout, lr, weight_decay, batch_size, n_hidden_layers, n_nodes = params
            n_epochs, batch_size, n_hidden_layers, n_nodes = (
                int(n_epochs),
                int(batch_size),
                int(n_hidden_layers),
                int(n_nodes),
            )

            print(
                f"***PARAMS n_epochs: {n_epochs}, dropout: {dropout}, lr: {lr}, weight_decay: {weight_decay}, batch_size: {batch_size}, n_hidden_layers {n_hidden_layers}, n_nodes {n_nodes}***"
            )

        auprc_list = []
        f1_list = []
        acc_list = []
        val_loss_list = []
        threshold_list = []
        self.n_epochs = []
        scores = []

        skf = StratifiedKFold(n_splits=5, shuffle=True)

        for k, (train_index, val_index) in enumerate(
            skf.split(self.X_train_clinical, self.Y_train_clinical)
        ):
            print(f"Kfold {k}")
            x_train_clinical = self.X_train_clinical[train_index]
            y_train_clinical = self.Y_train_clinical[train_index]
            x_train_cpet = self.X_train_cpet[train_index]
            y_train_cpet = self.Y_train_cpet[train_index]

            x_val_clinical = self.X_train_clinical[val_index]
            y_val_clinical = self.Y_train_clinical[val_index]
            x_val_cpet = self.X_train_cpet[val_index]
            y_val_cpet = self.Y_train_cpet[val_index]

            trainset = Model_1_2_dataset(
                x_train_clinical, y_train_clinical, x_train_cpet, y_train_cpet
            )
            trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
            valset = Model_1_2_dataset(x_val_clinical, y_val_clinical, x_val_cpet, y_val_cpet)
            valloader = DataLoader(valset, batch_size=len(valset))

            model_1 = Model_1().to(device)
            model_1.load_state_dict(
                torch.load(f"{MODEL_WEIGHTS}/Model_1.pth", map_location=device, weights_only=True)
            )
            model_1.eval()

            model_2 = Model_2().to(device)
            model_2.load_state_dict(
                torch.load(f"{MODEL_WEIGHTS}/Model_2.pth", map_location=device, weights_only=True)
            )
            model_2.eval()

            model = Model_1_2(dropout, n_hidden_layers, n_nodes).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            # Early stopping parameters
            last_val_loss = 999
            es_count = 0
            patience = 3
            delta = 1e-4
            stop_training = False

            pos_samples = y_train_clinical.sum()
            neg_samples = len(y_train_clinical) - pos_samples
            pos_weight = torch.tensor([neg_samples / pos_samples]).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            for e in range(n_epochs):
                # Training
                model.train()
                train_loss = 0
                val_loss = 0

                for batch in trainloader:
                    x_batch_clinical, y_batch, x_batch_cpet, _ = batch
                    x_batch_clinical, y_batch, x_batch_cpet = (
                        x_batch_clinical.float().to(device),
                        y_batch.to(device),
                        x_batch_cpet.float().to(device),
                    )

                    with torch.no_grad():
                        pred_clin, feat_clinical = model_1(x_batch_clinical)  # Clinical feature map
                        pred_cpet, feat_cpet = model_2(x_batch_cpet)  # CPET feature map

                    if self.use_pred:
                        y_pred = model(pred_clin, pred_cpet)
                        y_pred.flatten()
                    else:
                        feature_maps = torch.concatenate((feat_clinical, feat_cpet), axis=1)
                        y_pred = model(feature_maps)

                    loss = criterion(y_pred[0].flatten(), y_batch)
                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    train_loss += loss.item() / len(trainloader)

                # Evaluation
                model.eval()
                y_preds = torch.empty(0).to(device)
                y_batches = torch.empty(0).to(device)

                if torch.isnan(loss):
                    train_loss = 999

                with torch.no_grad():
                    for batch in valloader:
                        x_batch_clinical, y_batch, x_batch_cpet, _ = batch
                        x_batch_clinical, y_batch, x_batch_cpet = (
                            x_batch_clinical.float().to(device),
                            y_batch.to(device),
                            x_batch_cpet.float().to(device),
                        )

                        pred_clin, feat_clinical = model_1(x_batch_clinical)
                        pred_cpet, feat_cpet = model_2(x_batch_cpet)
                        feature_maps = torch.concatenate((feat_clinical, feat_cpet), axis=1)

                        if self.use_pred:
                            y_pred = model(pred_clin, pred_cpet)
                            y_pred.flatten()
                        else:
                            feature_maps = torch.concatenate((feat_clinical, feat_cpet), axis=1)
                            y_pred = model(feature_maps)

                        v_loss = criterion(y_pred[0].flatten(), y_batch.float())
                        val_loss += v_loss.item() / len(valloader)

                        y_pred = F.sigmoid(y_pred[0]).flatten()
                        y_preds = torch.cat((y_preds, y_pred), dim=0)
                        y_batches = torch.cat((y_batches, y_batch), dim=0)

                if torch.isnan(v_loss):
                    val_loss = 999

                print(f"Epoch {e + 1}/{n_epochs} - train loss {train_loss} - val loss {val_loss}")

                # Early stopping
                if (last_val_loss - val_loss) < delta:
                    es_count += 1
                    if es_count >= patience:
                        stop_training = True
                else:
                    es_count = 0

                if stop_training:
                    print(f"Early stop at {e + 1}")
                    break

                last_val_loss = val_loss

            self.n_epochs.append(e + 1)
            threshold = self.optimize_threshold(y_preds, y_batches, "F1")
            threshold_list.append(threshold)

            auprc = binary_auprc(y_preds, y_batches)
            f1 = binary_f1_score(y_preds, y_batches, threshold=threshold)
            acc = binary_accuracy(y_preds, y_batches, threshold=threshold)
            auprc_list.append(auprc.item())
            f1_list.append(f1.item())
            acc_list.append(acc.item())
            val_loss_list.append(last_val_loss)
            scores.append(last_val_loss - f1.item())

        score = np.mean(scores)

        print(
            f"AUCPR: {np.mean(auprc_list):.3f} ({np.std(auprc_list):.3f}), F1: {np.mean(f1_list):.3f} ({np.std(f1_list):.3f}), ACC: {np.mean(acc_list):.3f} ({np.std(acc_list):.3f}), val_loss {np.mean(val_loss_list):.4f} ({np.std(val_loss_list):.4f}), score {score:.4f} ({np.std(scores):.4f}), threshold {np.mean(threshold_list):.3f} ({np.std(threshold_list):.3f})"
        )

        if score < self.best_score:
            self.best_score = score
            self.threshold = np.mean(threshold_list)

        self.n_epochs = int(np.ceil(np.mean(self.n_epochs)))

        return score

    def run_bo_gp(self, bounds, max_iter):
        """Run the original Gaussian-process Bayesian search over GPyOpt bounds.

        max_iter controls search iterations. Returns [best_objective, parameters]
        with the selected threshold appended to parameters; records self.n_epochs."""
        self.threshold = None
        self.best_score = 999

        bo_gp = BayesianOptimization(
            f=self.objective_func, domain=bounds, acquisition_optimizer_type="CMA"
        )

        bo_gp.run_optimization(max_iter=max_iter)
        print(f"Best parameters found with score {bo_gp.fx_opt}:")
        print(bo_gp.x_opt)

        return [bo_gp.fx_opt, np.append(bo_gp.x_opt, self.threshold)]


class Optimizer_Model_1_3_4:
    """Bayesian search for Model-1-3-4.

    Construct with the training arrays only; validation folds are created by the
    objective. Training data is stored on the instance, not loaded from disk."""

    def __init__(
        self,
        model_name,
        X_train_clinical,
        Y_train_clinical,
        X_train_bbb,
        Y_train_bbb,
        X_train_ecg,
        Y_train_ecg,
    ):
        """Initialize Optimizer_Model_1_3_4 with the supplied study configuration."""
        self.threshold = None
        self.best_score = 999
        self.model_name = model_name
        self.X_train_clinical = X_train_clinical
        self.Y_train_clinical = Y_train_clinical
        self.X_train_bbb = X_train_bbb
        self.Y_train_bbb = Y_train_bbb
        self.X_train_ecg = X_train_ecg
        self.Y_train_ecg = Y_train_ecg

    def optimize_threshold(self, y_pred, y_true, metric):
        """Search the original 50-point [0, 1] grid for AP, F1, or recall.

        Returns the first improving threshold, or 0.0 when every score is zero.
        Metric names are case-sensitive; unsupported names raise ValueError."""
        threshold = 0.0  # First grid point is the deterministic fallback for all-zero scores.
        if metric not in {"AP", "F1", "recall"}:
            raise ValueError(f"Unsupported threshold metric: {metric}")
        if metric == "AP":
            ap_temp = 0
            for th in np.linspace(0, 1):
                y_pred_th = [1 if val > th else 0 for val in y_pred]
                temp = average_precision_score(y_true.cpu(), y_pred_th)
                if temp > ap_temp:
                    ap_temp = temp
                    threshold = th
        elif metric == "F1":
            f1_temp = 0
            for th in np.linspace(0, 1):
                temp = binary_f1_score(y_pred, y_true, threshold=th)
                if temp > f1_temp:
                    f1_temp = temp
                    threshold = th
        elif metric == "recall":
            recall_temp = 0
            for th in np.linspace(0, 1):
                y_pred_th = [1 if val > th else 0 for val in y_pred]
                temp = recall_score(y_true.cpu(), y_pred_th)
                if temp > recall_temp:
                    recall_temp = temp
                    threshold = th

        return threshold

    def objective_func(self, params):
        """Train the model across five shuffled stratified folds for one parameter row.

        params is the two-dimensional array supplied by GPyOpt. Returns the original
        scalar study objective and updates the best threshold and mean epoch count.
        Fold shuffling remains unseeded to preserve the existing research behavior."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        params = params[0]
        n_epochs, lr, weight_decay, batch_size = params
        n_epochs, batch_size = int(n_epochs), int(batch_size)

        print(
            f"***PARAMS n_epochs: {n_epochs}, lr: {lr}, weight_decay: {weight_decay}, batch_size: {batch_size}***"
        )

        auprc_list = []
        f1_list = []
        acc_list = []
        val_loss_list = []
        threshold_list = []
        self.n_epochs = []
        scores = []

        skf = StratifiedKFold(n_splits=5, shuffle=True)

        for k, (train_index, val_index) in enumerate(skf.split(self.X_train_bbb, self.Y_train_bbb)):
            print(f"Kfold {k}")
            x_train_clinical = [self.X_train_clinical[i] for i in train_index]
            y_train_clinical = [self.Y_train_bbb[i] for i in train_index]
            x_train_bbb = [self.X_train_bbb[i] for i in train_index]
            y_train_bbb = [self.Y_train_bbb[i] for i in train_index]
            x_train_ecg = [self.X_train_ecg[i] for i in train_index]
            y_train_ecg = [self.Y_train_ecg[i] for i in train_index]

            x_val_clinical = [self.X_train_clinical[i] for i in val_index]
            y_val_clinical = [self.Y_train_bbb[i] for i in val_index]
            x_val_bbb = [self.X_train_bbb[i] for i in val_index]
            y_val_bbb = [self.Y_train_bbb[i] for i in val_index]
            x_val_ecg = [self.X_train_ecg[i] for i in val_index]
            y_val_ecg = [self.Y_train_ecg[i] for i in val_index]

            trainset = Model_1_3_4_dataset(
                x_train_clinical,
                y_train_clinical,
                x_train_bbb,
                y_train_bbb,
                x_train_ecg,
                y_train_ecg,
            )
            trainloader = DataLoader(
                trainset, batch_size=batch_size, shuffle=True, collate_fn=collate_model_1_3_4
            )
            valset = Model_1_3_4_dataset(
                x_val_clinical, y_val_clinical, x_val_bbb, y_val_bbb, x_val_ecg, y_val_ecg
            )
            valloader = DataLoader(valset, batch_size=len(valset), collate_fn=collate_model_1_3_4)

            model_1 = Model_1().to(device)
            model_1.load_state_dict(
                torch.load(f"{MODEL_WEIGHTS}/Model_1.pth", map_location=device, weights_only=True)
            )
            model_1.eval()

            model_3 = Model_3().to(device)
            model_3.load_state_dict(
                torch.load(f"{MODEL_WEIGHTS}/Model_3.pth", map_location=device, weights_only=True)
            )
            model_3.eval()

            model_4 = Model_4().to(device)
            model_4.load_state_dict(
                torch.load(f"{MODEL_WEIGHTS}/Model_4.pth", map_location=device, weights_only=True)
            )
            model_4.eval()

            model = Model_1_3_4().to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            # Early stopping parameters
            last_val_loss = 999
            es_count = 0
            patience = 3
            delta = 1e-4
            stop_training = False

            pos_samples = np.sum(y_train_clinical)
            neg_samples = len(y_train_clinical) - pos_samples
            pos_weight = torch.tensor([neg_samples / pos_samples]).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            for e in range(n_epochs):
                # Training
                model.train()
                train_loss = 0
                val_loss = 0

                for batch in trainloader:
                    x_batch_clinical, y_batch, x_batch_bbb, _, x_batch_ecg, _, _ = batch
                    x_batch_clinical, y_batch, x_batch_bbb, x_batch_ecg = (
                        x_batch_clinical.float().to(device),
                        y_batch.to(device),
                        x_batch_bbb.float().to(device),
                        x_batch_ecg.float().to(device),
                    )

                    # Clinical & beat-by-beat
                    with torch.no_grad():
                        pred_clin, feat_clin = model_1(x_batch_clinical)
                        pred_ecg, feat_ecg = model_4(x_batch_ecg)

                    x_splitted = x_batch_bbb.unfold(dimension=1, size=30, step=15).to(device)
                    preds_bbb = torch.empty((0)).to(device)

                    for i in range(x_splitted.size(1)):
                        win = x_splitted[:, i, :, :]  # (batch, feats, win_len)

                        with torch.no_grad():
                            pred_bbb, feat_bbb, _, _ = model_3(win)

                        preds_bbb = torch.cat((preds_bbb, pred_bbb.unsqueeze(0)), dim=0)

                    pred_bbb = preds_bbb.mean(0).flatten()
                    y_pred = model(
                        pred_clin.flatten(), pred_bbb, pred_ecg.flatten()
                    )  # Weighted average of predictions

                    loss = criterion(y_pred.flatten(), y_batch.float())
                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    train_loss += loss.item() / len(trainloader)

                # Evaluation
                model.eval()
                y_preds = torch.empty(0).to(device)
                y_batches = torch.empty(0).to(device)

                if torch.isnan(loss):
                    train_loss = 999

                with torch.no_grad():
                    for batch in valloader:
                        x_batch_clinical, y_batch, x_batch_bbb, _, x_batch_ecg, _, _ = batch
                        x_batch_clinical, y_batch, x_batch_bbb, x_batch_ecg = (
                            x_batch_clinical.float().to(device),
                            y_batch.to(device),
                            x_batch_bbb.float().to(device),
                            x_batch_ecg.float().to(device),
                        )

                        # Clinical & beat-by-beat
                        pred_clin, feat_clin = model_1(x_batch_clinical)
                        pred_ecg, feat_ecg = model_4(x_batch_ecg)

                        x_splitted = x_batch_bbb.unfold(dimension=1, size=30, step=15).to(device)
                        preds_bbb = torch.empty((0)).to(device)

                        for i in range(x_splitted.size(1)):
                            win = x_splitted[:, i, :, :]  # (batch, feats, win_len)

                            with torch.no_grad():
                                pred_bbb, feat_bbb, _, _ = model_3(win)

                            preds_bbb = torch.cat((preds_bbb, pred_bbb.unsqueeze(0)), dim=0)

                        pred_bbb = preds_bbb.mean(0).flatten()
                        y_pred = model(
                            pred_clin.flatten(), pred_bbb, pred_ecg.flatten()
                        )  # Weighted average of predictions

                        v_loss = criterion(y_pred.flatten(), y_batch.float())
                        val_loss += v_loss.item() / len(valloader)

                        y_preds = torch.cat((y_preds, y_pred), dim=0)
                        y_batches = torch.cat((y_batches, y_batch), dim=0)

                if torch.isnan(v_loss):
                    val_loss = 999

                print(f"Epoch {e + 1}/{n_epochs} - train loss {train_loss} - val loss {val_loss}")

                # Early stopping
                if (last_val_loss - val_loss) < delta:
                    es_count += 1
                    if es_count >= patience:
                        stop_training = True
                else:
                    es_count = 0

                if stop_training:
                    print(f"Early stop at {e + 1}")
                    break

                last_val_loss = val_loss

            # Compute prediction mean across windows on validation set
            y_preds, y_batches = y_preds.flatten(), y_batches.flatten()

            self.n_epochs.append(e + 1)

            threshold = self.optimize_threshold(y_preds, y_batches, "F1")
            threshold_list.append(threshold)

            auprc = binary_auprc(y_preds, y_batches)
            f1 = binary_f1_score(y_preds, y_batches, threshold=threshold)
            acc = binary_accuracy(y_preds, y_batches, threshold=threshold)
            auprc_list.append(auprc.item())
            f1_list.append(f1.item())
            acc_list.append(acc.item())
            val_loss_list.append(last_val_loss)
            scores.append(last_val_loss - f1.item())

        score = np.mean(scores)

        print(
            f"AUCPR: {np.mean(auprc_list):.3f} ({np.std(auprc_list):.3f}), F1: {np.mean(f1_list):.3f} ({np.std(f1_list):.3f}), ACC: {np.mean(acc_list):.3f} ({np.std(acc_list):.3f}), val_loss {np.mean(val_loss_list):.4f} ({np.std(val_loss_list):.4f}), score {score:.4f} ({np.std(scores):.4f}), threshold {np.mean(threshold_list):.3f} ({np.std(threshold_list):.3f})"
        )

        if score < self.best_score:
            self.best_score = score
            self.threshold = np.mean(threshold_list)

        self.n_epochs = int(np.ceil(np.mean(self.n_epochs)))

        return score

    def run_bo_gp(self, bounds, max_iter, X_init=None, Y_init=None):
        """Run the original Gaussian-process Bayesian search over GPyOpt bounds.

        max_iter controls search iterations. Returns [best_objective, parameters]
        with the selected threshold appended to parameters; records self.n_epochs."""
        self.threshold = None
        self.best_score = 999

        bo_gp = BayesianOptimization(
            f=self.objective_func,
            domain=bounds,
            acquisition_optimizer_type="CMA",
            X=X_init,
            Y=Y_init,
        )

        bo_gp.run_optimization(max_iter=max_iter)
        print(f"Best parameters found with score {bo_gp.fx_opt}:")
        print(bo_gp.x_opt)

        return [bo_gp.fx_opt, np.append(bo_gp.x_opt, self.threshold)]
