import torch
import torch.nn as nn
import yaml

# Load config
with open("configs/eeg_config.yaml", "r") as f:
    config = yaml.safe_load(f)
    fusion_conf = config["fusion"]


class PhysioEncoder(nn.Module):
    def __init__(self):
        super(PhysioEncoder, self).__init__()

        self.input_features = 16  # As per prompt
        self.channels = 32
        self.embedding_dim = fusion_conf["embedding_dim"]

        # 1D CNN
        self.cnn = nn.Conv1d(self.input_features, 32, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

        # LSTM
        self.lstm = nn.LSTM(32, 128, batch_first=True)

        # Projection
        self.fc = nn.Linear(128, self.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [batch, features=16, timepoints]
        """
        x = self.cnn(x)
        x = self.relu(x)

        # Permute for LSTM [batch, seq, features]
        x = x.permute(0, 2, 1)

        _, (hn, _) = self.lstm(x)

        # Use last hidden state
        # hn: [layers, batch, hidden]
        embedding = self.fc(hn[-1])

        return embedding
