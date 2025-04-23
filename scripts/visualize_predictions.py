import torch
import matplotlib
matplotlib.use('Agg')  # Disable GUI and use file-only backend
import matplotlib.pyplot as plt
from PIL import Image  # <- This line is required to open the saved image
import os

from model_hetero_convlstm import HeteroConvLSTM

# Load model
model = HeteroConvLSTM()
model.load_state_dict(torch.load("outputs/hetero_convlstm.pth"))
model.eval()

# Dummy input (replace with your real sequence)
x = torch.rand(1, 7, 10, 10)  # Shape: (batch_size, time_steps, height, width)

# Run prediction
with torch.no_grad():
    pred = model(x).numpy()[0]

# Ensure output directory exists
os.makedirs("outputs", exist_ok=True)

# Plot and save the prediction
plt.imshow(pred, cmap='hot', interpolation='nearest')
plt.title("Predicted Accident Density")
plt.colorbar(label='Accident Count')
plt.savefig("outputs/prediction_heatmap.png")
plt.close()

# Open and display image
img = Image.open("outputs/prediction_heatmap.png")
img.show()
