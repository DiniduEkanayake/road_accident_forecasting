import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def simulate_grid_data(grid_size=(10, 10), days=60):
    rows = []
    for day in range(days):
        date = datetime(2023, 1, 1) + timedelta(days=day)
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                accidents = np.random.poisson(lam=np.random.uniform(0.1, 1.5))
                rows.append([x, y, date.strftime('%Y-%m-%d'), accidents])
    df = pd.DataFrame(rows, columns=["grid_x", "grid_y", "date", "accidents"])
    df.to_csv("data/simulated_accidents.csv", index=False)
    print("✅ Simulated grid accident data saved!")

if __name__ == "__main__":
    simulate_grid_data()