import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import os
from sklearn.metrics import mean_squared_error

# Set your data folder (same folder as the script)
data_folder = "./"

# Load data
confirmed = pd.read_csv(os.path.join(data_folder, "time_series_covid19_confirmed_global.csv"))
deaths = pd.read_csv(os.path.join(data_folder, "time_series_covid19_deaths_global.csv"))
recovered = pd.read_csv(os.path.join(data_folder, "time_series_covid19_recovered_global.csv"))

# Function to extract data for a specific country
def get_country_data(df, country):
    country_data = df[df['Country/Region'] == country]
    return country_data.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long']).sum()

# Select country
country = "India"

# Get data
confirmed_data = get_country_data(confirmed, country)
deaths_data = get_country_data(deaths, country)
recovered_data = get_country_data(recovered, country)

# Convert index (dates) to datetime
dates = pd.to_datetime(confirmed_data.index, format="%m/%d/%y")

# Build combined DataFrame
data = pd.DataFrame({
    'Date': dates,
    'Confirmed': confirmed_data.values,
    'Deaths': deaths_data.values,
    'Recovered': recovered_data.values
})

# Calculate additional columns
data['Infected'] = data['Confirmed'] - data['Deaths'] - data['Recovered']
data['Infected'] = data['Infected'].clip(lower=0)

# Assume fixed population
N = 1_400_000_000
data['Susceptible'] = N - data['Infected'] - data['Recovered'] - data['Deaths']
data['Exposed'] = 0  # Placeholder

# Filter for First Wave (March 15 – Nov 30, 2020)
first_wave = data[(data['Date'] >= '2020-03-15') & (data['Date'] <= '2020-11-30')].reset_index(drop=True)

# Define breakpoints (indices for 4 periods)
breaks = np.array_split(np.arange(len(first_wave)), 4)

# Define SEIR model with time-varying beta
def seir_model_timevarying(y, t, N, betas, sigma, gamma):
    S, E, I, R = y
    if t < breaks[0][-1]:
        beta = betas[0]
    elif t < breaks[1][-1]:
        beta = betas[1]
    elif t < breaks[2][-1]:
        beta = betas[2]
    else:
        beta = betas[3]
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

# Initial conditions
S0 = N - first_wave['Infected'][0] - first_wave['Recovered'][0] - first_wave['Deaths'][0]
E0 = 0
I0 = first_wave['Infected'][0]
R0 = first_wave['Recovered'][0]
y0 = [S0, E0, I0, R0]

# Time array
t = np.arange(len(first_wave))

# Solve SEIR model
def solve_seir(t, betas, sigma, gamma):
    solution = odeint(seir_model_timevarying, y0, t, args=(N, betas, sigma, gamma))
    S, E, I, R = solution.T
    return S, E, I, R

# Define fitting function
def fitting_func(t, beta1, beta2, beta3, beta4, sigma, gamma):
    betas = [beta1, beta2, beta3, beta4]
    _, _, I, R = solve_seir(t, betas, sigma, gamma)
    return np.concatenate([I, R])

# Data for curve fitting
ydata = np.concatenate([first_wave['Infected'].values, first_wave['Recovered'].values])

# Initial parameter guess
initial_guess = [0.3, 0.25, 0.2, 0.15, 1/5.2, 1/10]

# Perform curve fitting
popt, _ = curve_fit(fitting_func, t, ydata, p0=initial_guess, maxfev=10000)

# Extract optimized parameters
beta1, beta2, beta3, beta4, sigma_opt, gamma_opt = popt
print(f"\n✅ Optimized Parameters:\nβ1={beta1:.4f}, β2={beta2:.4f}, β3={beta3:.4f}, β4={beta4:.4f}, σ={sigma_opt:.4f}, γ={gamma_opt:.4f}")

# Get model predictions
S_fit, E_fit, I_fit, R_fit = solve_seir(t, [beta1, beta2, beta3, beta4], sigma_opt, gamma_opt)

# Clip negative values
S_fit = np.clip(S_fit, 0, None)
E_fit = np.clip(E_fit, 0, None)
I_fit = np.clip(I_fit, 0, None)
R_fit = np.clip(R_fit, 0, None)

# ------------------------------------------------------
# ✅ RMSE Calculation BEFORE dividing by 1 million
real_infected = first_wave['Infected'].values
real_recovered = first_wave['Recovered'].values

predicted_infected = I_fit
predicted_recovered = R_fit

rmse_infected = np.sqrt(mean_squared_error(real_infected, predicted_infected))
rmse_recovered = np.sqrt(mean_squared_error(real_recovered, predicted_recovered))

print(f"\n📈 RMSE (Infected Real vs Model): {rmse_infected/1_000_000:.4f} million people")
print(f"📈 RMSE (Recovered Real vs Model): {rmse_recovered/1_000_000:.4f} million people")
# ------------------------------------------------------

# Convert all outputs to millions
first_wave['Infected'] /= 1_000_000
first_wave['Recovered'] /= 1_000_000
first_wave['Deaths'] /= 1_000_000
I_fit /= 1_000_000
R_fit /= 1_000_000
S_fit /= 1_000_000
E_fit /= 1_000_000

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(first_wave['Date'], first_wave['Infected'], label='Real Infected', color='red')
plt.plot(first_wave['Date'], I_fit, label='Model Infected', linestyle='--', color='blue')
plt.plot(first_wave['Date'], first_wave['Recovered'], label='Real Recovered', color='green')
plt.plot(first_wave['Date'], R_fit, label='Model Recovered', linestyle='--', color='purple')
plt.xlabel('Date')
plt.ylabel('Number of People (in Millions)')
plt.title(f'COVID-19 First Wave in {country} (Time-Varying β SEIR Model Fitting)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save SEIR Fitted Data
seir_fitted = pd.DataFrame({
    'Date': first_wave['Date'],
    'Susceptible': S_fit,
    'Exposed': E_fit,
    'Infected': I_fit,
    'Recovered': R_fit,
    'Deaths': first_wave['Deaths']
})

seir_fitted.to_csv(os.path.join(data_folder, "seir_fitted_data.csv"), index=False)
print("\n✅ SEIR fitted data saved to 'seir_fitted_data.csv'")
