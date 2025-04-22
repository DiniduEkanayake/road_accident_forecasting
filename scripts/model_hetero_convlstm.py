import torch
import torch.nn as nn

class HeteroConvLSTM(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=64, output_size=100):  # 10x10 = 100
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grid_size = 10  # if using 10x10 grids

        self.lstm = nn.LSTM(input_size=self.grid_size * self.grid_size,
                            hidden_size=hidden_dim,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)  # Predict flat grid

    def forward(self, x):
        B, T, H, W = x.size()
        x = x.view(B, T, -1)              # [B, T, H*W]
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]     # Last timestep → [B, hidden]
        out = self.fc(lstm_out)           # [B, 100]
        return out.view(B, H, W)          # Reshape to [B, 10, 10]
