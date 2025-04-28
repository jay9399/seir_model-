import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import os

# Load SEIR model fitted data
data_folder = "./"
seir_data = pd.read_csv(os.path.join(data_folder, "seir_fitted_data.csv"))

# Use only First Wave dates
first_wave = seir_data[(seir_data['Date'] >= '2020-03-15') & (seir_data['Date'] <= '2020-11-30')].reset_index(drop=True)

# Build input matrix: ['Susceptible', 'Exposed', 'Infected', 'Recovered']
results = first_wave[['Susceptible', 'Exposed', 'Infected', 'Recovered']].values
T = len(results)

# Prepare data for ML (sliding window)
look_back = 7  # Past 7 days
forecast_horizon = 1  # Now only predict 1 day ahead!

X, y = [], []
for i in range(T - look_back - forecast_horizon + 1):
    x_seq = results[i:i+look_back].flatten()
    y_seq = results[i+look_back:i+look_back+forecast_horizon].flatten()
    X.append(x_seq)
    y.append(y_seq)

X, y = np.array(X), np.array(y)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train MLP Regressor
mlp = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

# --- Recursive Forecasting ---

# Start with the last available 7 days from the first_wave
initial_input = results[-look_back:].flatten()

predictions = []

# Predict recursively 30 days ahead
steps_ahead = 30
for step in range(steps_ahead):
    next_pred = mlp.predict(initial_input.reshape(1, -1))[0]
    predictions.append(next_pred)

    # Update input: remove oldest day (4 features), add newly predicted day
    initial_input = np.concatenate([initial_input[4:], next_pred.flatten()])

# Format predictions
predictions = np.array(predictions).reshape(steps_ahead, 4)

# For comparison: true future data (after training data)
true_future = results[-(steps_ahead):]

# Evaluate
mse = mean_squared_error(true_future, predictions)
print(f"\nMean Squared Error (Recursive Forecast, 30 days): {mse:.2f}\n")

# Plot True vs Predicted
labels = ['Susceptible', 'Exposed', 'Infected', 'Recovered']
colors = ['blue', 'orange', 'red', 'green']

plt.figure(figsize=(14, 10))

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.plot(true_future[:, i]/1e6, label=f"True {labels[i]}", color=colors[i], linestyle='--')
    plt.plot(predictions[:, i]/1e6, label=f"Predicted {labels[i]}", color=colors[i])
    plt.xlabel("Days Ahead")
    plt.ylabel("Population (in Millions)")
    plt.title(labels[i])
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.suptitle("COVID-19 First Wave Forecast in India (MLP Regressor with Recursive Prediction)", fontsize=18, y=1.02)
plt.show()
