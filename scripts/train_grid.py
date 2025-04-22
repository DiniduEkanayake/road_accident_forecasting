import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model_hetero_convlstm import HeteroConvLSTM

B, T, H, W = 16, 7, 10, 10
x = torch.rand(B, T, H, W)
y_true = torch.rand(B, H, W)

model = HeteroConvLSTM()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y_true)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "outputs/hetero_convlstm.pth")