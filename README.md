# Road Accident Forecasting Sample

This project demonstrates a spatio-temporal deep learning approach to forecast road accidents using simulated data on a 10×10 spatial grid. The system uses a Hetero-ConvLSTM model trained with synthetic data, optimized using MSE loss, and visualized via heatmaps.

---

## 📁 Data

- **`data/simulated_accidents.csv`**  
  Contains synthetic road accident data generated over 60 days for a 10×10 spatial grid (each cell ~1 km²).

---

## 📜 Scripts

### 1. `generate_data.py`
- Simulates accident data for each grid cell over 60 days.
- Uses Poisson distribution (λ between 0.1 and 1.5) for accident count.
- Stores (x, y, date, accidents) in a CSV file.
- Output: `data/simulated_accidents.csv`
- Executable standalone for easy integration.

---

### 2. `model_hetero_convlstm.py`
- Defines the **Hetero-ConvLSTM** architecture.
- Inputs: Flattened sequences of 10×10 grids over time.
- Core: LSTM layer for temporal modeling + Linear layer for prediction.
- Outputs: Reconstructed 10×10 accident risk grid.
- Designed for import (not directly executable).

---

### 3. `train_grid.py`
- Simulates 7-day input + 1-day target tensors.
- Trains `HeteroConvLSTM` model using:
  - **Loss**: Mean Squared Error (MSE)
  - **Optimizer**: Adam (lr = 0.001)
  - **Epochs**: 20
- Logs training loss per epoch.
- Saves model to `outputs/hetero_convlstm.pth`.

#### 🔍 Mean Squared Error (MSE) Loss:
- Measures average squared difference between actual and predicted values.
- Punishes larger errors more.
- Common in regression tasks like temperature or stock price prediction.

#### 🔁 Training Epochs:
- 1 epoch = 1 full pass through dataset.
- Multiple epochs improve learning, but too many can cause overfitting.

---

### 4. `visualize_predictions.py`
- Loads the trained model from `outputs/hetero_convlstm.pth`.
- Simulates a random 7-day accident data input.
- Generates a forecast and visualizes it using Matplotlib.
- Saves prediction heatmap as `predicted_heatmap.png`.

---

## 📊 Output

- Forecast visualized as a heatmap:
  - Grid intensity indicates predicted accident risk.
  - Helps identify spatial accident patterns.

---

## 🧠 Model Summary

- **Type**: Hetero-ConvLSTM (LSTM + Fully Connected)
- **Input Shape**: `[Batch, Time, 10, 10]`
- **Output Shape**: `[Batch, 10, 10]`
- **Task**: Spatio-temporal forecasting (accident risk)

---


