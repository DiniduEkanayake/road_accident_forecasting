import torch
import matplotlib.pyplot as plt
from model_hetero_convlstm import HeteroConvLSTM

model = HeteroConvLSTM()
model.load_state_dict(torch.load("outputs/hetero_convlstm.pth"))
model.eval()

x = torch.rand(1, 7, 10, 10)
pred = model(x).detach().numpy()[0]

plt.imshow(pred, cmap='hot', interpolation='nearest')
plt.colorbar(label='Predicted Accident Risk')
plt.title("Forecasted Accident Heatmap")
plt.savefig("outputs/predicted_heatmap.png")
plt.show()