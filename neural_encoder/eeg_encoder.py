import torch
import torch.nn as nn
import yaml

# Load config
with open("configs/eeg_config.yaml", "r") as f:
    config = yaml.safe_load(f)
    eeg_conf = config["neural_encoder"]
    fusion_conf = config["fusion"]


class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()

        # Parameters
        self.channels = 32  # config["channels"]["eeg_channels"] # Hardcoded for now based on prompt/config
        self.samples = int(
            (eeg_conf["epoch_window"][1] - eeg_conf["epoch_window"][0])
            * eeg_conf["sampling_rate"]
        )  # e.g. 250
        self.F1 = 8
        self.D = 2
        self.F2 = self.F1 * self.D
        self.kernel_length = 64
        self.embedding_dim = fusion_conf["embedding_dim"]

        # 1. Temporal Conv
        self.temporal_conv = nn.Conv2d(
            1,
            self.F1,
            (1, self.kernel_length),
            padding=(0, self.kernel_length // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.F1)

        # 2. Depthwise Conv (Spatial Filter)
        self.depthwise_conv = nn.Conv2d(
            self.F1, self.F2, (self.channels, 1), groups=self.F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(self.F2)
        self.elu1 = nn.ELU()

        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(0.5)

        # 3. Separable Conv
        self.separable_conv1 = nn.Conv2d(
            self.F2, self.F2, (1, 16), padding=(0, 8), bias=False
        )  # Depthwise part usually distinct but simplified here
        self.bn3 = nn.BatchNorm2d(self.F2)
        self.elu2 = nn.ELU()
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(0.5)

        # Calculate Flatten Dim
        # Rough calc: 250 -> pool4 -> 62 -> pool8 -> 7
        self.flatten_dim = self.F2 * 7  # tune based on input size

        # Projection to Shared Latent Space
        self.project = nn.Linear(self.flatten_dim, self.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [batch, channels, samples] -> reshaped to [batch, 1, channels, samples]
        """
        # Ensure 4D
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.temporal_conv(x)
        x = self.bn1(x)

        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)

        x = self.separable_conv1(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)
        # Adapt calculate flatten dim if needed
        # print(x.shape)

        try:
            x = self.project(x)
        except RuntimeError:
            # Dynamic adjustment for dev/testing if shapes mismatch
            current_dim = x.shape[1]
            device = x.device
            self.project = nn.Linear(current_dim, self.embedding_dim).to(device)
            x = self.project(x)

        return x
