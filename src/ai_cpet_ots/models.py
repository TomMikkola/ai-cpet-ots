"""Six checkpoint-compatible PyTorch models for OTS classification.

Model numbers follow the study: clinical (1), conventional CPET (2),
breath-by-breath (3), ECG (4), feature fusion (1-2), and probability fusion
(1-3-4). Thresholds are study defaults; labels are control=0 and OTS=1."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model_1(nn.Module):
    """Clinical MLP for (batch, 4) inputs.

    Returns logits (batch, 1) and dense features (batch, n_nodes). Constructor
    arguments retain study defaults, including the 0.563 decision threshold."""

    def __init__(self, dropout=4e-1, n_hidden_layers=3, n_nodes=128, threshold=0.563):
        """Initialize Model_1 with the supplied study configuration."""
        super(Model_1, self).__init__()

        self.threshold = threshold

        layers = nn.ModuleList([])

        # First layer
        first_layer = nn.Sequential(
            nn.Linear(4, n_nodes), nn.BatchNorm1d(n_nodes), nn.ReLU(), nn.Dropout(dropout)
        )

        layers.append(first_layer)

        # Hidden layers
        for _ in range(n_hidden_layers):
            hidden_layer = nn.Sequential(
                nn.Linear(n_nodes, n_nodes), nn.BatchNorm1d(n_nodes), nn.ReLU(), nn.Dropout(dropout)
            )

            layers.append(hidden_layer)

        self.dense = nn.Sequential(*layers)
        self.output = nn.Linear(n_nodes, 1)

    def forward(self, x):
        """Apply the layers and return the outputs documented on Model_1."""
        dense_output = self.dense(x)
        output = self.output(dense_output)
        return output, dense_output


class Model_2(nn.Module):
    """Conventional-CPET MLP for (batch, 11) inputs.

    Returns logits (batch, 1) and dense features (batch, final_width). Hidden
    widths halve to a minimum of 16; the default decision threshold is 0.547."""

    def __init__(self, dropout=0, n_hidden_layers=4, n_nodes=16, threshold=0.547):
        """Initialize Model_2 with the supplied study configuration."""
        super(Model_2, self).__init__()

        self.threshold = threshold
        self.limit = 16

        layers = nn.ModuleList([])

        # First layer
        first_layer = nn.Sequential(
            nn.Linear(11, n_nodes * n_hidden_layers),
            nn.BatchNorm1d(n_nodes * n_hidden_layers),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        layers.append(first_layer)

        # Hidden layers
        last_n_nodes = n_nodes * n_hidden_layers

        for _ in range(n_hidden_layers):
            next_n_nodes = last_n_nodes // 2 if last_n_nodes // 2 >= self.limit else self.limit

            hidden_layer = nn.Sequential(
                nn.Linear(last_n_nodes, next_n_nodes),
                nn.BatchNorm1d(next_n_nodes),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            layers.append(hidden_layer)
            last_n_nodes = next_n_nodes

        self.dense = nn.Sequential(*layers)
        self.output = nn.Linear(last_n_nodes, 1)

    def forward(self, x):
        """Apply the layers and return the outputs documented on Model_2."""
        dense_output = self.dense(x)
        output = self.output(dense_output)
        return output, dense_output


class Model_3(nn.Module):
    """Attention CNN for breath-by-breath windows (batch, 14, window_length).

    Returns logits, 64 dense features, feature attention, and temporal attention.
    The study uses 30-second windows at one sample per second and threshold 0.784."""

    def __init__(self, dropout=0, threshold=0.784):
        """Initialize Model_3 with the supplied study configuration."""
        super(Model_3, self).__init__()

        self.dropout = dropout
        self.threshold = threshold

        # 1st layer
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 5), padding="same"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        # 2nd layer
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 5), padding="same"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        # 3rd layer
        self.block3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 5), padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
        )

        # 4th layer
        self.block4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 3), padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # 5th layer
        self.block5 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 3), padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # 6th layer
        self.block6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 3), padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(14, 1)), nn.BatchNorm2d(64), nn.ReLU()
        )

        self.dense = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, 1),
        )

        self.feat_att = nn.Sequential(nn.Linear(14, 3), nn.ReLU(), nn.Linear(3, 14), nn.Sigmoid())

        self.W2 = nn.Linear(64, 64, bias=False)
        self.U2 = nn.Linear(64, 1, bias=False)

    def compute_feat_weights(self, x_batch):
        """Return channel attention of shape (batch, 1, 14) from time-major windows."""
        feat_means = x_batch.mean(dim=1)

        # Pass to channel based attention mechanism
        att_weights = self.feat_att(feat_means)  # (batch, feats)

        return att_weights.unsqueeze(1)  # (batch, 1, feats)

    def compute_temp_weights(self, output):
        # Compute temp weights
        """Return softmax attention across time for (batch, time, 64) CNN features."""
        tanh_out = torch.tanh(self.W2(output))  # (batch, seq_len, 1)
        att_scores = self.U2(tanh_out)  # (batch, seq_len, 1)
        temp_weights = F.softmax(att_scores, dim=1)  # (batch, seq_len, 1)
        return temp_weights

    def forward(self, x_batch):
        """
        x_batch     (batch, features, seg_len)
        output      (batch, 1)
        """
        # Compute feature attention weights
        feat_att_weights = self.compute_feat_weights(x_batch.permute(0, 2, 1))  # (batch, 1, feats)
        x_batch = x_batch * feat_att_weights.permute(0, 2, 1)  # (batch, features, segment_length)

        x_batch = x_batch.unsqueeze(1)  # (batch_size, 1, features, seg_len)

        cnn_output = self.block1(x_batch)
        cnn_output = self.block2(cnn_output)
        cnn_output = self.block3(cnn_output)
        cnn_output = self.block4(cnn_output)
        cnn_output = self.block5(cnn_output)
        cnn_output = self.block6(cnn_output)
        cnn_output = self.spatial(cnn_output)  # (batch, 64, 1, reduced_time)
        cnn_output = cnn_output.squeeze(2)  # (batch, feats, red_window_size)

        # Compute temporal attention weights
        temp_att_weights = self.compute_temp_weights(cnn_output.permute(0, 2, 1))
        cnn_output = torch.sum(
            temp_att_weights * cnn_output.permute(0, 2, 1), dim=1
        )  # (batch, feats)

        output = self.dense(cnn_output)

        return output, cnn_output, feat_att_weights, temp_att_weights


class Model_4(nn.Module):
    """ECG CNN for (batch, sequence_length) input sampled at 500 Hz.

    Uses 15000-sample (30-second) windows and returns logits plus 9600 pooled
    features. The fixed dense layer requires the original window configuration.
    The default decision threshold is 0.322."""

    def __init__(self, dropout=0.8, threshold=0.322, win_size=15000, step_size=15000):
        """Initialize Model_4 with the supplied study configuration."""
        super(Model_4, self).__init__()

        self.dropout = dropout
        self.threshold = threshold
        self.win_size = win_size
        self.step_size = step_size

        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=500, stride=5, padding=249),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=500, stride=5, padding=249),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(self.dropout),
            nn.MaxPool1d(2),
        )
        self.dense = nn.Sequential(nn.Linear(9600, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, x_batch):
        """
        x_batch     (batch, seq_len)
        output      (batch, 1)
        """

        outputs = []  # (n_win, batch, feats, win_size)

        # Split to windows
        x_batch_unfolded = x_batch.unfold(
            dimension=1, size=self.win_size, step=self.step_size
        )  # (batch, n_win, win_size)

        for win_i in range(x_batch_unfolded.size(1)):
            win = x_batch_unfolded[:, win_i, :].unsqueeze(1)  # (batch, 1, win_size)
            cnn_output = self.net(win)

            outputs.append(cnn_output)

        outputs_stacked = (
            torch.stack(outputs).permute(1, 0, 3, 2).mean(1).flatten(1, -1)
        )  # (batch, red_win_size*feat)
        output = self.dense(outputs_stacked)

        return output, outputs_stacked


class Model_1_2(nn.Module):
    """Feature-fusion MLP for concatenated clinical and CPET features.

    Input shape is (batch, 144), comprising 128 clinical and 16 CPET features.
    Returns logits and dense features; the default decision threshold is 0.722."""

    def __init__(self, dropout=6e-1, n_hidden_layers=1, n_nodes=128, threshold=0.722):
        """Initialize Model_1_2 with the supplied study configuration."""
        super(Model_1_2, self).__init__()

        self.threshold = threshold

        layers = nn.ModuleList([])

        # First layer
        first_layer = nn.Sequential(
            nn.Linear(144, n_nodes), nn.BatchNorm1d(n_nodes), nn.ReLU(), nn.Dropout(dropout)
        )

        layers.append(first_layer)

        # Hidden layers
        last_n_nodes = n_nodes
        for _ in range(n_hidden_layers):
            hidden_layer = nn.Sequential(
                nn.Linear(last_n_nodes, last_n_nodes * 2),
                nn.BatchNorm1d(last_n_nodes * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            layers.append(hidden_layer)
            last_n_nodes *= 2

        self.dense = nn.Sequential(*layers)
        self.output = nn.Linear(last_n_nodes, 1)

    def forward(self, x):
        """Apply the layers and return the outputs documented on Model_1_2."""
        dense_output = self.dense(x)
        output = self.output(dense_output)
        return output, dense_output


class Model_1_3_4(nn.Module):
    """Learned softmax-weighted fusion of three base-model logits.

    Converts clinical, breath-by-breath, and ECG logits to probabilities before
    fusion. Returns probabilities directly (do not apply another sigmoid).
    The default decision threshold is 0.290."""

    def __init__(self, threshold=0.290):
        """Initialize Model_1_3_4 with the supplied study configuration."""
        super(Model_1_3_4, self).__init__()

        self.threshold = threshold

        self.w1 = nn.Parameter(torch.randn(1))
        self.w2 = nn.Parameter(torch.randn(1))
        self.w3 = nn.Parameter(torch.randn(1))

    def forward(self, pred_clin, pred_bbb, pred_ecg):
        """Apply the layers and return the outputs documented on Model_1_3_4."""
        weights = torch.softmax(torch.cat([self.w1, self.w2, self.w3]), dim=0)

        pred_clin = F.sigmoid(pred_clin)
        pred_bbb = F.sigmoid(pred_bbb)
        pred_ecg = F.sigmoid(pred_ecg)

        output = weights[0] * pred_clin + weights[1] * pred_bbb + weights[2] * pred_ecg

        return output
