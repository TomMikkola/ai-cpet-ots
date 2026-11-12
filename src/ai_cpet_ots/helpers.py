"""Signal preprocessing, variable-length batching, and diagnostic plots.

Sampling rates, window lengths, filter coefficients, and normalization formulas
follow the original study implementation; they have not been recalibrated."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
from scipy.signal import lfilter
from torch.nn.utils.rnn import pad_sequence


def get_predictions(model, model_1, model_3, loader):
    """Return feature-fusion probabilities for every batch in loader order.

    model_1 and the legacy-named model_3 argument supply clinical/CPET feature
    maps. All three models enter evaluation mode; an empty loader raises ValueError."""
    predictions = []

    device = next(model.parameters()).device
    model.eval()
    model_1.eval()
    model_3.eval()
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
            pred_cpet, feat_cpet = model_3(x_batch_cpet)

            feature_maps = torch.concatenate((feat_clinical, feat_cpet), axis=1)
            y_pred, _ = model(feature_maps)

            predictions.append(F.sigmoid(y_pred).flatten())

    if not predictions:
        raise ValueError("Prediction loader must contain at least one batch.")
    return torch.cat(predictions)


def segment_BBB_data(X, Y=None, window_size=30, overlap=15):
    """Split (time, features) BBB tensors into labeled windows.

    Returns stacked (windows, features, window_size) tensors, repeated labels,
    and window counts per subject. The legacy overlap argument is used as the
    unfold stride; padding assumes 14 features. Y is required by the study path."""
    n_feats = X[0].size(-1)
    X_train_splitted = torch.empty((0, n_feats, window_size))
    Y_train_splitted = torch.empty((0, 1))
    seg_lengths = []

    for i, x in enumerate(X):
        x_splitted = x.unfold(dimension=0, size=window_size, step=overlap)

        # Add padding to last window if not equal to window_Size
        n_windows = x.size(0) // (window_size - (window_size - overlap))

        if x_splitted.size(0) < n_windows:
            last_idx = int(
                (x_splitted.size(0) * x_splitted.size(-1))
                // (window_size / (window_size - overlap))
            )
            padding = torch.zeros((1, 14, window_size - (x.size(0) - last_idx)))
            last_data = x[last_idx:].permute(1, 0).unsqueeze(0)
            last_win = torch.cat((last_data, padding), dim=2)
            x_splitted = torch.cat((x_splitted, last_win), dim=0)

        seg_lengths.append(x_splitted.size(0))
        X_train_splitted = torch.vstack((X_train_splitted, x_splitted))
        Y_for_split = torch.tensor(Y[i]).repeat(x_splitted.size(0), 1)
        Y_train_splitted = torch.vstack((Y_train_splitted, Y_for_split))

    print(
        f"Created {len(X_train_splitted)} segments with window size {window_size} and overlap {overlap}"
    )

    return X_train_splitted, Y_train_splitted, seg_lengths


def pan_tompkins(signal):
    # Define Low-Pass Filter (LPF) Coefficients
    """Apply the original high/low-pass coefficients and roll out 21-sample delay.

    Accepts a NumPy-compatible signal and returns a NumPy array. SciPy filters
    operate along the last axis; coefficients and boundary behavior are unchanged."""
    b_lpf = [36, 0, 0, 0, 0, -72, 0, 0, 0, 0, 36]  # Numerator
    a_lpf = [1, -2, 1]  # Denominator

    # Define High-Pass Filter (HPF) Coefficients
    b_hpf = [-1 / 32] + [0] * 15 + [1, -1] + [0] * 14 + [1 / 32]  # Numerator (33 coefficients)
    a_hpf = [1, -1]  # Denominator

    # Apply the High-Pass Filter (HPF)
    x_hpf = lfilter(b_hpf, a_hpf, signal)

    # Apply the Low-Pass Filter (LPF)
    x_lpf = lfilter(b_lpf, a_lpf, x_hpf)

    lp_delay = (len(b_lpf) - 1) // 2  # 5 samples
    hp_delay = (len(b_hpf) - 1) // 2  # 16 samples
    total_delay = lp_delay + hp_delay  # 21 samples
    x_lpf = np.roll(x_lpf, -total_delay)

    return x_lpf


def compute_mean_sd_curves(signal, win_size):
    """Return a uniform-filter mean curve and the original covariance-based SD.

    win_size is measured in samples. The returned SD is a scalar, not a moving SD."""
    mean = uniform_filter1d(signal, win_size)
    sd = np.sqrt(np.cov(signal, signal))[0, 0]

    return mean, sd


def compute_median_curve(signal, win_size, step_size):
    """Interpolate medians of stepped signal windows back to sample positions.

    win_size and step_size are in samples; extrapolation follows the study code."""
    medians = []
    n_windows = int(np.ceil(len(signal) / (win_size - (win_size - step_size))))  # + 1

    last_win = 0
    for _ in range(n_windows):
        win = signal[last_win : last_win + win_size - (win_size - step_size)]
        medians.append(np.median(win))
        last_win += win_size - (win_size - step_size)

    interp_fun = interp1d(
        np.arange(0, len(signal), win_size - (win_size - step_size)),
        medians,
        fill_value="extrapolate",
    )
    median = interp_fun(np.arange(0, len(signal)))
    return median


def trim_spikes(f):
    """Clip ECG spikes using the original asymmetric SD and median-envelope rules.

    Accepts a NumPy array or CPU tensor and returns a NumPy array of the original
    length. Fixed windows assume a 500 Hz signal; formulas are unchanged."""
    win_size = 30000  # 60s
    step_size = int(win_size / 2)  # 30s
    try:
        f = f.numpy()
    except (AttributeError, TypeError, ValueError):
        pass

    # Trim very large spikes
    mean, sd = compute_mean_sd_curves(f, 15000)
    pos_idxs = np.where(f > mean + 8 * sd)
    neg_idxs = np.where(f < mean - 5 * sd)

    while len(pos_idxs[0]) > 0 or len(neg_idxs[0]) > 0:
        f[pos_idxs] = mean[pos_idxs] + 8 * sd
        f[neg_idxs] = mean[neg_idxs] - 5 * sd
        # Update
        mean, sd = compute_mean_sd_curves(f, win_size)
        pos_idxs = np.where(f > mean + 8 * sd)
        neg_idxs = np.where(f < mean - 5 * sd)

    max_env = maximum_filter1d(f, 7500)
    max_env = np.append(max_env, np.zeros(win_size) + np.mean(max_env))

    min_env = minimum_filter1d(f, 7500)
    min_env = np.append(min_env, np.zeros(win_size) - np.mean(max_env))

    max_env_median = compute_median_curve(max_env, win_size, step_size)
    min_env_median = compute_median_curve(min_env, win_size, step_size)

    f = np.append(f, np.zeros(win_size))

    idxs = np.where(f > max_env_median * 1.1)
    while len(idxs[0]) > 0:
        f[idxs] = max_env_median[idxs] * 1.0

        max_env = maximum_filter1d(f, 7500)
        max_env[-win_size:] = np.mean(max_env)
        max_env_median = compute_median_curve(max_env, win_size, step_size)

        idxs = np.where(f > max_env_median * 1.1)

    idxs = np.where(f < min_env_median * 1.1)
    while len(idxs[0]) > 0:
        f[idxs] = min_env_median[idxs] * 1.0

        min_env = minimum_filter1d(f, 7500)
        min_env[-win_size:] = np.mean(min_env)
        min_env_median = compute_median_curve(min_env, win_size, step_size)

        idxs = np.where(f < min_env_median * 1.1)

    f = f[:-win_size]

    return f


def normalize(signal):
    """Nurmaini2020
    signal      (seg_len, channel)
    """
    try:
        ub = np.array([1, 1])
        lb = np.array([0, 0])

        coef = (
            (ub - lb) / (signal.max(dim=1).values.numpy() - signal.min(dim=1).values.numpy())
        ).reshape(2, 1)
        x_mid = (
            (
                signal.max(dim=1).values - (signal.max(dim=1).values - signal.min(dim=1).values) / 2
            ).numpy()
        ).reshape(2, 1)
        mid = (ub - (ub - lb) / 2).reshape(2, 1)

        return signal.numpy() * coef - (x_mid * coef) + mid
    except (AttributeError, TypeError, ValueError):
        try:
            ub = np.array([1, 1])
            lb = np.array([0, 0])

            coef = ((ub - lb) / (np.max(signal, axis=0) - np.min(signal, axis=0))).reshape(1, 2)
            x_mid = (
                np.max(signal, axis=0) - (np.max(signal, axis=0) - np.min(signal, axis=0)) / 2
            ).reshape(1, 2)
            mid = (ub - (ub - lb) / 2).reshape(1, 2)

            return signal * coef - (x_mid * coef) + mid
        except (AttributeError, TypeError, ValueError):
            ub = 1
            lb = 0

            coef = (ub - lb) / (signal.max() - signal.min())
            x_mid = signal.max() - (signal.max() - signal.min()) / 2
            mid = ub - (ub - lb) / 2

            return signal * coef - (x_mid * coef) + mid


def collate_model_1_3_4(batch):
    """Pad aligned BBB/ECG sequences while stacking clinical features and labels.

    Returns the same seven fields as Model_1_3_4_dataset, with sequence lists
    replaced by batch-first zero-padded tensors and original BBB lengths."""
    X_clinical, Y_clinical, X_bbb, Y_bbb, X_ecg, Y_ecg, seq_lengths = zip(*batch)
    X_bbb = pad_sequence(X_bbb, batch_first=True)
    X_ecg = pad_sequence(X_ecg, batch_first=True)

    return (
        torch.stack(X_clinical),
        torch.stack(Y_clinical),
        X_bbb,
        torch.stack(Y_bbb),
        X_ecg,
        torch.stack(Y_ecg),
        seq_lengths,
    )


def collate(batch):
    """Zero-pad variable-length samples into (batch, time, ...) tensors.

    Returns padded sequences, the label tuple, and the original length tuple."""
    sequences, labels, seq_lengths = zip(*batch)
    batch = pad_sequence(sequences, batch_first=True)

    return batch, labels, seq_lengths


def preds_to_bin(model, model_preds):
    """Threshold model probabilities into Python labels (0=control, 1=OTS)."""
    return [1 if val >= model.threshold else 0 for val in model_preds]


def metrics(y_true, y_pred):
    """Return sensitivity, specificity, and likelihood ratios for binary labels.

    Inputs are array-like and use control=0, OTS=1. Undefined ratios retain the
    original NumPy/inf behavior; this function does not estimate confidence intervals."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # May become inf/undefined if specificity = 1
    lr_pos = sensitivity / (1 - specificity) if specificity < 1 else np.inf

    lr_neg = (1 - sensitivity) / specificity if specificity > 0 else np.inf

    return {
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "LR_plus": lr_pos,
        "LR_minus": lr_neg,
    }


def Fagan(draw=[], ax=None, labels=[]):
    """Draw the original Fagan nomogram and return its Matplotlib axes.

    The study caller supplies four (pretest_probability, likelihood_ratio, label)
    entries and legend labels. Annotations are in English.
    Plotting constants and probability transforms are unchanged."""
    Pmax = 0.99
    Pmin = 0.01
    Omax = np.ceil(np.log(Pmax / (1 - Pmax)))
    Omin = np.floor(np.log(Pmin / (1 - Pmin)))

    Pticks = np.sort(
        np.concatenate(
            [
                np.arange(0.1, 0.999, 0.1),
                np.arange(0.02, 0.1, 0.02),
                np.arange(0.9, 0.99, 0.02),
                [0.01, 0.99],
            ]
        )
    )

    Oticks = Pticks / (1 - Pticks)

    Lticks = [10 ** int(i) for i in np.arange(-3, 3.1, 1)]

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4), dpi=1200)
        ax.set_xlim(-1.2, 1.2)

    # LEFT
    ax.axvline(-1, c="k", linewidth=1)
    ax.scatter(
        [-1 for _ in range(len(Pticks))], np.log(Oticks), marker="_", c="k", s=100, linewidth=1
    )
    for Otick, Ptick in zip(Oticks, Pticks):
        ax.text(
            -1.05,
            np.log(Otick),
            f"${Ptick * 100:.0f}$" if Ptick <= 0.99 and Ptick >= 0.01 else f"${Ptick * 100:.1f}$",
            va="center",
            ha="right",
        )

    # RIGHT
    ax.axvline(+1, c="k", linewidth=1)
    ax.scatter(
        [+1 for _ in range(len(Pticks))], np.log(1 / Oticks), marker="_", c="k", s=100, linewidth=1
    )
    for Otick, Ptick in zip(Oticks, Pticks):
        ax.text(
            +1.05,
            np.log(Otick),
            f"${(1 - Ptick) * 100:.0f}$"
            if Ptick <= 0.99 and Ptick >= 0.01
            else f"${(1 - Ptick) * 100:.1f}$",
            va="center",
            ha="left",
        )
    ax.set_ylim(Pmin, Pmax)

    # CENTER
    Ops = []
    for Ltick in Lticks:
        Op = 1 / (np.sqrt(Ltick) + 1)
        Ops.append(Op)
        ax.scatter(0, np.log(Op / (1 - Op)), marker="_", c="k", s=100, linewidth=1)
        ax.text(+0.05, np.log(Op / (1 - Op)), f"${Ltick}$", c="k", va="center", ha="left")
        ax.text(-0.05, np.log(Op / (1 - Op)), f"${Ltick}$", c="k", va="center", ha="right")
    ax.plot(
        [0, 0],
        [np.log(Ops[0] / (1 - Ops[0])), np.log(Ops[-1] / (1 - Ops[-1]))],
        c="k",
        ls="-",
        linewidth=1,
    )
    ax.set_ylim(Omax, Omin)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title("Fagan's nomogram", fontsize=12)
    ax.text(-1, 5.5, "Pre-test\nprobability %", ha="center", va="center", fontsize=12, rotation=0)
    ax.text(0, 5.5, "Likelihood\nratio", ha="center", va="center", fontsize=12, rotation=0)
    ax.text(+1, 5.5, "Post-test\nprobability %", ha="center", va="center", fontsize=12, rotation=0)

    colors = []
    styles = []
    style_list = ["solid", "dashed"] * 2

    for i in range(len(draw)):
        if i >= len(draw) // 2:  # 2nd column
            idx = i - len(draw) // 2
            colors.append(plt.cm.tab10(idx))
        else:
            colors.append(plt.cm.tab10(i))
        styles.append(style_list[i])

    handles = [None] * (len(draw))

    colors[0] = "black"
    colors[1] = "black"
    colors[2] = "black"
    colors[3] = "black"

    for i, line in enumerate(draw):
        (line_handle,) = ax.plot(
            [-1, 1],
            [np.log(line[0] / (1 - line[0])), np.log((1 - line[0]) / (line[0] * line[1]))],
            label=line[-1],
            color=colors[i],
            linestyle=styles[i],
            linewidth=1,
        )
        handles[i] = line_handle

    # Create custom thicker handles just for the legend
    legend_handles = [
        Line2D([0], [0], color="black", linestyle=styles[i], linewidth=1) for i in range(len(draw))
    ]

    plt.legend(
        legend_handles,
        labels,
        loc="lower center",
    )
    return ax
