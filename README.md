# AQI_Prediction_Using_Prophet

Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.

Prophet is open source software released by Facebook’s Core Data Science team. It is available for download on CRAN and PyPI.

    So, Prophet is the facebooks’ open source tool for making time series predictions.

    Prophet decomposes time series data into trend, seasonality and holiday effect.

    Trend models non periodic changes in the time series data.

    Seasonality is caused due to the periodic changes like daily, weekly, or yearly seasonality.

    Holiday effect which occur on irregular schedules over a day or a period of days.

    Error terms is what is not explained by the model.

2. Advantages of Prophet ¶

Table of Contents

Prophet has several advantages associated with it. These are given below:-

    1. Accurate and fast - Prophet is accurate and fast. It is used in many applications across Facebook for producing reliable forecasts for planning and goal setting.

    2. Fully automatic - Prophet is fully automatic. We will get a reasonable forecast on messy data with no manual effort.

    3. Tunable forecasts - Prophet produces adjustable forecasts. It includes many possibilities for users to tweak and adjust forecasts. We can use human-interpretable parameters to improve the forecast by adding our domain knowledge.

    4. Available in R or Python - We can implement the Prophet procedure in R or Python.

    5. Handles seasonal variations well - Prophet accommodates seasonality with multiple periods.

    6. Robust to outliers - It is robust to outliers. It handles outliers by removing them.

    7. Robust to missing data - Prophet is resilient to missing data.





Detailed Code Explanation & Report: Air Quality Forecasting Using Prophet
Objective
The provided Python code is a comprehensive workflow that:

Loads and preprocesses air quality data from an Excel file.
Scales the data using Min-Max Scaling.
Trains a forecasting model (Prophet) for three air quality parameters: Minimum AQI, Maximum AQI, and Average AQI.
Forecasts future AQI values.
Evaluates predictions using Mean Absolute Error (MAE) and Mean Squared Error (MSE).
Visualizes the forecasted results.
The main goal is to predict AQI values for future days based on historical data.

1. Imported Libraries

    import pandas as pd                      # Data manipulation and analysis
    import numpy as np                       # Numerical computations
    from prophet import Prophet              # Forecasting model (time series)
    import matplotlib.pyplot as plt          # Plotting graphs
    from sklearn.preprocessing import MinMaxScaler  # Data normalization
    from sklearn.metrics import mean_absolute_error, mean_squared_error  # Evaluation metrics
2. Function-Level Explanation
   
    2.1. load_and_preprocess_data(file_path)
    Purpose: Loads the data from an Excel file and preprocesses it by converting non-numeric values into numeric, then interpolating missing values.
    
    Steps:
    
    Reads Excel file into a pandas DataFrame.
    Converts the 'Date' column to datetime format.
    Converts the AQI columns ('Minimum AQI', 'Maximum AQI', 'Average AQI') to numeric. Invalid parsing results in NaN.
    Fills missing values using linear interpolation.
    Returns: Cleaned data and a list of AQI column names.
    
    2.2. scale_data(data, columns)
    Purpose: Normalizes selected columns using Min-Max scaling (values between 0 and 1).
    
    Steps:
    
    Instantiates MinMaxScaler.
    Fits and transforms the specified columns (AQI data).
    Returns: Scaled data and the scaler object.
    
    2.3. prepare_prophet_df(data, column_name)
    Purpose: Prepares the data for Prophet by renaming columns to ds (date) and y (target variable).
    
    Returns: A DataFrame with two columns: ds and y.
    
    2.4. train_prophet_model(df)
    Purpose: Instantiates and fits a Prophet model to the provided time series data.
    
    Returns: A trained Prophet model.
    
    2.5. forecast_future(model, periods)
    Purpose: Uses the trained Prophet model to make future predictions.
    
    Steps:
    
    Generates future dates for the next n_days.
    Predicts future AQI values for those dates.
    Returns: Forecast DataFrame with predicted values.
    
    2.6. plot_forecast(model, forecast, param)
    Purpose: Plots the forecasted AQI values.
    
    Steps:
    
    Uses Prophet’s built-in plot function to visualize the predictions.
    Customizes the plot with title and labels.
   
4. forecast_air_quality(file_path, n_days)
    The main function that runs the entire pipeline.
    
    Steps:
    Load and preprocess the data
    Calls load_and_preprocess_data() and returns cleaned data and column names.
    
    Train-test split
    
    Uses the last 10 records as test_data.
    Uses the rest of the data as train_data.
    Scale the training data and test data
    
    Scales training data (scaled_train_data) and applies the same scaling to test data (scaled_test_data).
    Forecasting for each AQI parameter (Minimum AQI, Maximum AQI, Average AQI):
    
    Prepares data for Prophet.
    Trains the Prophet model.
    Forecasts the next n_days.
    Extracts predicted AQI values from the forecast (yhat column).
    Compares predicted values to actual values from the test dataset.
    Calculates error metrics:
    Mean Absolute Error (MAE)
    Mean Squared Error (MSE)
    Prints the error metrics for each AQI parameter.
    Returns:
    forecast_arrays: NumPy arrays of forecasted AQI values for each parameter.
    test_arrays: NumPy arrays of actual AQI test values for comparison.
   
4. Example Usage

    file_path = "Aqi_result.xlsx"
    n_days = 10
    forecast_arrays, test_arrays = forecast_air_quality(file_path, n_days)
    file_path: Input Excel file containing AQI data with columns Date, Minimum AQI, Maximum AQI, Average AQI.
    n_days: Number of days to forecast (10 days).
    Outputs forecast_arrays (predictions) and test_arrays (actual data).
    5. Key Features & Functionality
    Feature	Description
    Prophet Modeling	Uses Facebook Prophet for time series forecasting, which accounts for seasonality and trends.
    MinMax Scaling	Normalizes the data to improve model performance.
    Error Metrics	Uses MAE and MSE to evaluate the accuracy of the forecast.
    Visualization	Plots the predicted AQI for easy interpretation (optional in the current function).

6. Suggestions for Improvement
    Inverse Transform Forecasts
    
    Currently, forecasts are in scaled format (0 to 1).
    To make forecasts meaningful, apply the inverse transformation:
    
    forecast_original = scaler.inverse_transform(forecast_arrays[param].reshape(-1, 1))
    Automated Plotting
    
    Call plot_forecast() inside the loop for visual analysis of each parameter.
    More Robust Evaluation
    
    Use R² Score or other metrics.
    Cross-validation instead of a simple train-test split.
    Hyperparameter Tuning (Prophet)
    
    Customize Prophet with parameters like seasonality_mode, changepoint_prior_scale.
    Add Logging
    
    Print model progress, execution time, etc.
   
7. Example Output (Conceptual)

    Parameter: Minimum AQI
    Mean Absolute Error: 0.08
    Mean Squared Error: 0.01
    
    Parameter: Maximum AQI
    Mean Absolute Error: 0.12
    Mean Squared Error: 0.02
    
    Parameter: Average AQI
    Mean Absolute Error: 0.10
    Mean Squared Error: 0.015
    Final Report Summary
    Purpose: To forecast air quality index (AQI) parameters using historical data and time series forecasting techniques.
    Technique Used: Prophet Model for time series forecasting.
    Preprocessing: Interpolation and Min-Max scaling.
    Evaluation: Compared actual and predicted values using MAE and MSE.

