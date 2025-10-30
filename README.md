# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 30.10.2025

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
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
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load the data
df_cars = pd.read_csv("cars (1).csv")

# Set 'Car' column as index (assuming 'Car' can serve as a time-like index or is a unique identifier)
# This is done because ARIMA models are typically used with time series data,
# and 'Car' column here is used as an index in lieu of a proper time index.
df_cars.set_index('Car', inplace=True)


# Define the ARIMA model function for a given target variable
def forecast_arima(dataframe, target_variable, order):
    """
    Fits an ARIMA model to the target variable in the dataframe,
    makes predictions, and visualizes the results.

    Args:
        dataframe (pd.DataFrame): The input dataframe with 'Car' as index.
        target_variable (str): The name of the column to forecast.
        order (tuple): The (p, d, q) order of the ARIMA model.
    """
    # Split data into training and testing sets
    train_size = int(len(dataframe) * 0.8)
    train_data, test_data = dataframe[:train_size], dataframe[train_size:]

    # Fit the ARIMA model
    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()

    # Make predictions
    predictions = fitted_model.forecast(steps=len(test_data))

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(test_data[target_variable], predictions))

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data')
    plt.plot(test_data.index, predictions, label='Forecasted Data')
    plt.xlabel('Car Index') # Changed xlabel to 'Car Index'
    plt.ylabel(target_variable)
    plt.title(f'ARIMA Forecasting for {target_variable}') # Use f-string for title
    plt.legend()
    plt.show()

    print(f"Root Mean Squared Error (RMSE) for {target_variable}: {rmse:.2f}") # Format RMSE to 2 decimal places


# Run the ARIMA model on the 'CO2' column with order (5, 1, 0)
# The order (p, d, q) represents:
# p: The number of lag observations (autoregressive order)
# d: The number of times that the raw observations are differenced (integrated order)
# q: The size of the moving average window (moving average order)
forecast_arima(df_cars, 'CO2', order=(5, 1, 0))
```
### OUTPUT:
<img width="850" height="547" alt="image" src="https://github.com/user-attachments/assets/047be56a-7410-4f23-8f48-217ff00f485b" />
Root Mean Squared Error (RMSE) for CO2: 9.46

### RESULT:
Thus the program run successfully based on the ARIMA model using python.
