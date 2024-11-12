# EX.NO.09        A project on Time series analysis on daily-minimum-temperatures using ARIMA model 
### Date: 

### NAME:RAGUNATH R
### REGISTER NO: 212222240081

### AIM:
To Create a project on Time series analysis on daily minimum temperatures using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of daily-minimum-temperatures
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from pmdarima import auto_arima  # Make sure pmdarima is installed: !pip install pmdarima

# Load the dataset with the correct column names
file_path = '/content/daily-minimum-temperatures-in-me.csv'
df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
df.columns = ['temperature']  # Rename for simplicity

# Convert the 'temperature' column to numeric, coercing errors to NaN
df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')

# Drop rows with NaN values in the 'temperature' column
df = df.dropna()

# Display the first few rows and statistical summary
print(df.head())
print(df.describe())

# Plot the time series data
plt.figure(figsize=(12, 6))
plt.plot(df['temperature'], label='Temperature')
plt.title('Time Series Plot of Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# Augmented Dickey-Fuller Test to check stationarity
def adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] < 0.05:
        print("The series is stationary.")
    else:
        print("The series is not stationary.")

adf_test(df['temperature'])

# Plot ACF and PACF to identify potential AR and MA terms
plot_acf(df['temperature'], lags=30)
plot_pacf(df['temperature'], lags=30)
plt.show()

# Differencing the series to make it stationary if necessary
df['temperature_diff'] = df['temperature'].diff().dropna()

plt.figure(figsize=(12, 6))
plt.plot(df['temperature_diff'], label='Differenced Temperature')
plt.title('Differenced Time Series')
plt.xlabel('Date')
plt.ylabel('Differenced Temperature')
plt.legend()
plt.show()

# Re-run the ADF test on the differenced data to check stationarity
adf_test(df['temperature_diff'].dropna())

# Define ARIMA order parameters, p, d, and q
# Typically determined by analyzing ACF/PACF plots and differencing
p, d, q = 1, 1, 1  # Example values; you may adjust them based on your analysis

# Fit the ARIMA model
model = ARIMA(df['temperature'], order=(p, d, q))
model_fit = model.fit()

# Print the model summary
print(model_fit.summary())

# Forecasting future values
forecast_steps = 10
forecast = model_fit.forecast(steps=forecast_steps)
print("Forecasted values:")
print(forecast)

# Plot original data and forecasted values
plt.figure(figsize=(12, 6))
plt.plot(df['temperature'], label='Original Data')
plt.plot(forecast.index, forecast, color='red', label='Forecast', marker='o')
plt.title('ARIMA Model Forecast')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# Using auto_arima to determine optimal (p, d, q) parameters
auto_model = auto_arima(df['temperature'], seasonal=False, stepwise=True, trace=True)
print(auto_model.summary())

# Train-Test Split for Forecast Evaluation
train_size = int(len(df) * 0.8)
train, test = df['temperature'][:train_size], df['temperature'][train_size:]

# Fit model on training data and evaluate on test data
train_model = ARIMA(train, order=(p, d, q))
train_model_fit = train_model.fit()
test_forecast = train_model_fit.forecast(steps=len(test))

# Calculate RMSE
rmse = sqrt(mean_squared_error(test, test_forecast))
print(f'RMSE on test set: {rmse}')

# Plot train, test, and forecasted values
plt.figure(figsize=(12, 6))
plt.plot(train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, test_forecast, label='Forecast', color='red')
plt.title('Train/Test Split with Forecast')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.show()
```

### OUTPUT:

![image](https://github.com/user-attachments/assets/65863274-58a4-4cfe-8168-e03484d1a779)
![image](https://github.com/user-attachments/assets/17752a0e-5a54-411e-b833-d3c9b57b9c52)
![image](https://github.com/user-attachments/assets/3affe450-d89b-4851-932b-a954fc5af4bf)
![image](https://github.com/user-attachments/assets/8bb6e7b0-b82f-4d71-a58c-1c4314bd64c2)
![image](https://github.com/user-attachments/assets/74ae1ff8-5b53-402b-be23-8812b4383c95)
![image](https://github.com/user-attachments/assets/76ea8d83-4ecc-41f9-8841-e4343d3c7272)
![image](https://github.com/user-attachments/assets/d50d6b94-fbd8-42ad-b9a7-2770f1a98080)
![image](https://github.com/user-attachments/assets/94ecd7ee-1fef-4cc1-9f5e-de156e03b03d)






### RESULT:
Thus the program run successfully based on the ARIMA model using python.
