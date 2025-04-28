import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import os

# Set your data folder
data_folder = "./"

# Load the SEIR fitted data
seir_fitted = pd.read_csv(os.path.join(data_folder, "seir_fitted_data.csv"))

# Ensure 'Date' is datetime
seir_fitted['Date'] = pd.to_datetime(seir_fitted['Date'])

# Parameters
window_size = 7
future_days = 15  # Forecast only 15 days

# Prepare windowed data
def prepare_window_data(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

# Use raw series directly (NO log transform)
infected_series = seir_fitted['Infected'].values
recovered_series = seir_fitted['Recovered'].values

X_infected, y_infected = prepare_window_data(infected_series, window_size)
X_recovered, y_recovered = prepare_window_data(recovered_series, window_size)

# Split into train/test
split_idx = int(0.8 * len(X_infected))
X_infected_train, X_infected_test = X_infected[:split_idx], X_infected[split_idx:]
y_infected_train, y_infected_test = y_infected[:split_idx], y_infected[split_idx:]

X_recovered_train, X_recovered_test = X_recovered[:split_idx], X_recovered[split_idx:]
y_recovered_train, y_recovered_test = y_recovered[:split_idx], y_recovered[split_idx:]

# Train Ridge Regression models
model_infected = Ridge(alpha=1.0)
model_infected.fit(X_infected_train, y_infected_train)

model_recovered = Ridge(alpha=1.0)
model_recovered.fit(X_recovered_train, y_recovered_train)

# Evaluate RMSE
y_infected_pred_test = model_infected.predict(X_infected_test)
y_recovered_pred_test = model_recovered.predict(X_recovered_test)

infected_rmse = np.sqrt(mean_squared_error(y_infected_test, y_infected_pred_test))
recovered_rmse = np.sqrt(mean_squared_error(y_recovered_test, y_recovered_pred_test))

# Evaluate R² score
infected_r2 = r2_score(y_infected_test, y_infected_pred_test)
recovered_r2 = r2_score(y_recovered_test, y_recovered_pred_test)

print(f"✅ Test RMSE - Infected: {infected_rmse:.6f} million")
print(f"✅ Test RMSE - Recovered: {recovered_rmse:.6f} million")
print(f"✅ Test R² - Infected: {infected_r2:.4f}")
print(f"✅ Test R² - Recovered: {recovered_r2:.4f}")

# Historical prediction (whole data)
y_infected_pred_all = model_infected.predict(X_infected)
y_recovered_pred_all = model_recovered.predict(X_recovered)

# Future Forecast
last_window_infected = infected_series[-window_size:].tolist()
last_window_recovered = recovered_series[-window_size:].tolist()

infected_preds_future = []
recovered_preds_future = []

for _ in range(future_days):
    next_infected = model_infected.predict([last_window_infected])[0]
    infected_preds_future.append(next_infected)
    last_window_infected.pop(0)
    last_window_infected.append(next_infected)

    next_recovered = model_recovered.predict([last_window_recovered])[0]
    recovered_preds_future.append(next_recovered)
    last_window_recovered.pop(0)
    last_window_recovered.append(next_recovered)

# Dates
future_dates = pd.date_range(start=seir_fitted['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_days)

# Plotting
plt.figure(figsize=(14, 7))
# SEIR historical
plt.plot(seir_fitted['Date'], seir_fitted['Infected'], label='SEIR Infected', color='red')
plt.plot(seir_fitted['Date'], seir_fitted['Recovered'], label='SEIR Recovered', color='green')
# ML historical prediction
plt.plot(seir_fitted['Date'][window_size:], y_infected_pred_all, '--', label='ML Predicted Infected (Historical)', color='orange')
plt.plot(seir_fitted['Date'][window_size:], y_recovered_pred_all, '--', label='ML Predicted Recovered (Historical)', color='blue')
# ML future prediction
plt.plot(future_dates, infected_preds_future, '--', label='ML Predicted Infected (Future)', color='darkorange')
plt.plot(future_dates, recovered_preds_future, '--', label='ML Predicted Recovered (Future)', color='deepskyblue')

# Add RMSE and R² to the plot
plt.text(0.05, 0.95, f"Test RMSE (Infected): {infected_rmse:.6f} million", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', color='orange')
plt.text(0.05, 0.90, f"Test RMSE (Recovered): {recovered_rmse:.6f} million", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', color='blue')
plt.text(0.05, 0.85, f"Test R² (Infected): {infected_r2:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', color='orange')
plt.text(0.05, 0.80, f"Test R² (Recovered): {recovered_r2:.4f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', color='blue')

plt.xlabel('Date')
plt.ylabel('Number of People (in Millions)')
plt.title('COVID-19 Forecasting (Historical + 15-Day Future) - Ridge Regression')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save future forecast
future_forecast = pd.DataFrame({
    'Date': future_dates,
    'Predicted Infected': infected_preds_future,
    'Predicted Recovered': recovered_preds_future
})
future_forecast.to_csv(os.path.join(data_folder, "future_forecast_seir_ml_ridge_normal.csv"), index=False)
print("\n✅ Future forecast saved to 'future_forecast_seir_ml_ridge_normal.csv'")
