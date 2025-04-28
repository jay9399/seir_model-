import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Set data folder (where your CSVs are)
data_folder = "./"

# Load the datasets
confirmed = pd.read_csv(os.path.join(data_folder, "time_series_covid19_confirmed_global.csv"))
deaths = pd.read_csv(os.path.join(data_folder, "time_series_covid19_deaths_global.csv"))
recovered = pd.read_csv(os.path.join(data_folder, "time_series_covid19_recovered_global.csv"))

# Function to extract country's data
def get_country_data(df, country):
    country_data = df[df['Country/Region'] == country]
    return country_data.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long']).sum()

# Select country
country = "India"

# Extract country data
confirmed_data = get_country_data(confirmed, country)
deaths_data = get_country_data(deaths, country)
recovered_data = get_country_data(recovered, country)

# Convert dates
dates = pd.to_datetime(confirmed_data.index, format="%m/%d/%y")

# Create DataFrame
data = pd.DataFrame({
    'Date': dates,
    'Confirmed': confirmed_data.values,
    'Deaths': deaths_data.values,
    'Recovered': recovered_data.values
})

# Calculate Infected and Susceptible
data['Infected'] = data['Confirmed'] - data['Deaths'] - data['Recovered']
data['Infected'] = data['Infected'].clip(lower=0)

# Assume total population
N = 1_400_000_000
data['Susceptible'] = N - data['Infected'] - data['Recovered'] - data['Deaths']

# Placeholder for Exposed
data['Exposed'] = 0

# Focus on First Wave (March 15 – Nov 30, 2020)
first_wave = data[(data['Date'] >= '2020-03-15') & (data['Date'] <= '2020-11-30')].reset_index(drop=True)

# Build the input matrix
results = first_wave[['Susceptible', 'Exposed', 'Infected', 'Recovered']].values
T = len(results)

# Prepare data for ML (sliding window)
look_back = 7
forecast_horizon = 30

X, y = [], []
for i in range(T - look_back - forecast_horizon):
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

# Predict on test set
y_pred = mlp.predict(X_test)

# Evaluate
idx = 0  # Select first test example
true_seq = y_test[idx].reshape(forecast_horizon, 4)
pred_seq = y_pred[idx].reshape(forecast_horizon, 4)

# Mean Squared Error
mse = mean_squared_error(true_seq, pred_seq)
print(f"\nMean Squared Error on selected forecast: {mse:.2f}\n")

# Plot True vs Predicted
labels = ['Susceptible', 'Exposed', 'Infected', 'Recovered']
colors = ['blue', 'orange', 'red', 'green']

plt.figure(figsize=(12, 8))

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.plot(true_seq[:, i], label=f"True {labels[i]}", color=colors[i], linestyle='--')
    plt.plot(pred_seq[:, i], label=f"Predicted {labels[i]}", color=colors[i])
    plt.xlabel("Days Ahead")
    plt.ylabel("Population")
    plt.title(labels[i])
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.suptitle(f"COVID-19 First Wave Prediction in {country} (MLP Regressor Forecast)", fontsize=16, y=1.02)
plt.show()