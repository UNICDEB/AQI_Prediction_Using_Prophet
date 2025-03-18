
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    try:
        data = pd.read_excel(file_path)
        data['Date'] = pd.to_datetime(data['Date'])
        # Convert non-numeric values to NaN and interpolate
        columns_to_convert = ['o3_Min', 'o3_Max', 'no2_Min', 'no2_Max', 'so2_Min', 'so2_Max', 'co_Min', 'co_Max', 
                            'pm1_Min', 'pm1_Max', 'pm25_Min', 'pm25_Max', 'pm10_Min', 'pm10_Max', 'nh3_Min', 'nh3_Max']
        for col in columns_to_convert:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        data = data.interpolate()
        return data, columns_to_convert
    except Exception as e:
        print(f"An error occurred in (load_and_preprocess_data) function: {e}")

# Function to scale data
def scale_data(data, columns):
    try:
        scaler = MinMaxScaler()
        scaled_data = data.copy()
        scaled_data[columns] = scaler.fit_transform(data[columns])
        return scaled_data, scaler
    except Exception as e:
        print(f"An error occurred in (scale_data) function: {e}")

# Function to prepare data for Prophet
def prepare_prophet_df(data, column_name):
    try:
        df = data[['Date', column_name]].rename(columns={'Date': 'ds', column_name: 'y'})
        return df
    except Exception as e:
        print(f"An error occurred in (prepare_prophet_df) function: {e}")

# Function to train Prophet model
def train_prophet_model(df):
    try:
        model = Prophet()
        model.fit(df)
        return model
    except Exception as e:
        print(f"An error occurred in (train_prophet_model) function: {e}")

# Function to forecast future values
def forecast_future(model, periods):
    try:
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return forecast
    except Exception as e:
        print(f"An error occurred in (forecast_future) function: {e}")

# Function to plot forecast
def plot_forecast(model, forecast, param):
    try:
        fig = model.plot(forecast)
        plt.title(f'Forecast for {param}')
        plt.xlabel('Date')
        plt.ylabel(param)
        plt.show()
    except Exception as e:
        print(f"An error occurred in (plot_forecast) function: {e}")

# Function to handle the entire workflow
def forecast_air_quality(file_path, n_days):
    try:
        # Load and preprocess data
        data, columns_to_convert = load_and_preprocess_data(file_path) 
        # Scale data
        # scaled_data, scaler = scale_data(data, columns_to_convert)
        scaled_data = data  
        # Define parameters
        parameters = columns_to_convert
        forecasts = {}   
        # Dictionary to store NumPy arrays for each parameter
        forecast_arrays = {}

        # Forecast each parameter
        for param in parameters:
            df = prepare_prophet_df(scaled_data, param)
            model = train_prophet_model(df)
            forecast = forecast_future(model, n_days)
            forecasts[param] = forecast
            
            # Extract forecast values for the desired number of days
            forecast_values = forecast[['ds', 'yhat_lower', 'yhat_upper', 'yhat']].tail(n_days)
            
            # Convert forecast values to a NumPy array and store it
            forecast_array = forecast_values.to_numpy()
            forecast_arrays[param] = forecast_array
            
            # Print the last few rows of forecast
            # print(f"\n\nPrediction of {param} is \n {forecast_values}")
            # print(f"NumPy array for {param} forecast values: \n {forecast_array}")

        return forecast_arrays
    except Exception as e:
        print(f"An error occurred in (forcast_airquality) function: {e}")



# PM10 SUB INDEX CALCULATION
def calculate_pm10_category(pm10):
    if not isinstance(pm10, str):
        if pm10 <= 50:
            return pm10
        elif 50 < pm10 <= 100:
            return pm10
        elif 100 < pm10 <= 250:
            return 100 + (pm10 - 100) * 100 / 150
        elif 250 < pm10 <= 350:
            return 200 + (pm10 - 250)
        elif 350 < pm10 <= 430:
            return 300 + (pm10 - 350) * (100 / 80)
        elif pm10 > 430:
            return 400 + (pm10 - 430) * (100 / 80)
    else:
        return 0

# PM2.5 SUB INDEX CALCULATION
def calculate_pm25_category(pm25):
    if not isinstance(pm25, str):
        if pm25 <= 30:
            return pm25 * 50 / 30
        elif 30 < pm25 <= 60:
            return 50 + (pm25 - 30) * 50 / 30
        elif 60 < pm25 <= 90:
            return 100 + (pm25 - 60) * 100 / 30
        elif 90 < pm25 <= 120:
            return 200 + (pm25 - 90) * (100 / 30)
        elif 120 < pm25 <= 250:
            return 300 + (pm25 - 120) * (100 / 130)
        elif pm25 > 250:
            return 400 + (pm25 - 250) * (100 / 130)
    else:
        return 0
    
# SO2 SUB INDEX CALCULATION
def calculate_so2_category(so2):
    if not isinstance(so2, str):
        if so2 <= 40:
            return so2 * 50 / 40
        elif 40 < so2 <= 80:
            return 50 + (so2 - 40) * 50 / 40
        elif 80 < so2 <= 380:
            return 100 + (so2 - 80) * 100 / 300
        elif 380 < so2 <= 800:
            return 200 + (so2 - 380) * (100 / 420)
        elif 800 < so2 <= 1600:
            return 300 + (so2 - 800) * (100 / 800)
        elif so2 > 1600:
            return 400 + (so2 - 1600) * (100 / 800)
    else:
        return 0
    
# NO2 SUB INDEX CALCULATION
def calculate_no2_category(no2):
    if not isinstance(no2, str):
        if no2 <= 40:
            return no2 * 50 / 40
        elif 40 < no2 <= 80:
            return 50 + (no2 - 40) * 50 / 40
        elif 80 < no2 <= 180:
            return 100 + (no2 - 80) * 100 / 100
        elif 180 < no2 <= 280:
            return 200 + (no2 - 180) * (100 / 100)
        elif 280 < no2 <= 400:
            return 300 + (no2 - 280) * (100 / 120)
        elif no2 > 400:
            return 400 + (no2 - 400) * (100 / 120)
    else:
        return 0
    
# CO SUB INDEX CALCULATION
def calculate_co_category(co):
    if not isinstance(co, str):
        if co <= 1:
            return co * 50 / 1
        elif 1 < co <= 2:
            return 50 + (co - 1) * 50 / 1
        elif 2 < co <= 10:
            return 100 + (co - 2) * 100 / 8
        elif 10 < co <= 17:
            return 200 + (co - 10) * (100 / 7)
        elif 17 < co <= 34:
            return 300 + (co - 17) * (100 / 17)
        elif co > 34:
            return 400 + (co - 34) * (100 / 17)
    else:
        return 0
# O3 SUB INDEX CALCULATION
def calculate_o3_category(o3):
    if not isinstance(o3, str):
        if o3 <= 50:
            return o3 * 50 / 50
        elif 50 < o3 <= 100:
            return 50 + (o3 - 50) * 50 / 50
        elif 100 < o3 <= 168:
            return 100 + (o3 - 100) * 100 / 68
        elif 168 < o3 <= 208:
            return 200 + (o3 - 168) * (100 / 40)
        elif 208 < o3 <= 748:
            return 300 + (o3 - 208) * (100 / 539)
        elif o3 > 748:
            return 400 + (o3 - 400) * (100 / 539)
    else:
        return 0
    
# NH3 SUB INDEX CALCULATION
def calculate_nh3_category(nh3):
    try:

        if not isinstance(nh3, str):
            if nh3 <= 200:
                return nh3 * 50 / 200
            elif 200 < nh3 <= 400:
                return 50 + (nh3 - 200) * 50 / 200
            elif 400 < nh3 <= 800:
                return 100 + (nh3 - 400) * 100 / 400
            elif 800 < nh3 <= 1200:
                return 200 + (nh3 - 800) * (100 / 400)
            elif 1200 < nh3 <= 1800:
                return 300 + (nh3 - 1200) * (100 / 600)
            elif nh3 > 1800:
                return 400 + (nh3 - 1800) * (100 / 600)
        else:
            return 0
    except Exception as e:
        print(f"An error occurred: {e}")
    
# AQI CALCULATOR
def aqi_calculator(pm10_result, pm25_result, so2_result,no2_result,co_result,o3_result,nh3_result):
    try:
        aqi = -float('inf')
        if(pm10_result>=1):
            aqi = pm10_result
        if(pm25_result>=1):
            aqi = pm25_result if (pm25_result>aqi) else aqi
        if(so2_result>=1):
            aqi = so2_result if (so2_result>aqi) else aqi
        if(no2_result>=1):
            aqi = no2_result if (no2_result>aqi) else aqi
        if(co_result>=1):
            aqi = co_result if (co_result>aqi) else aqi
        if(o3_result>=1):
            aqi = o3_result if (o3_result>aqi) else aqi
        if(nh3_result>=1):
            aqi = nh3_result if (nh3_result>aqi) else aqi
    except Exception as e:
        print("Error in AQI calculation: ", str(e))
             
    return(aqi)

def calculate_min_max_values(i, forecast_arrays):
    pollutants = ['o3', 'no2', 'so2', 'co', 'pm1', 'pm25', 'pm10', 'nh3']
    min_values = {}
    max_values = {}

    for pollutant in pollutants:
        min_key = f'{pollutant}_Min'
        max_key = f'{pollutant}_Max'
        
        min_value = forecast_arrays[min_key][i][1]
        max_value = forecast_arrays[max_key][i][2]
        
        min_values[f'{pollutant}_MIN'] = min(min_value, max_value)
        max_values[f'{pollutant}_MAX'] = max(min_value, max_value)

    # print("Minimim Value - ", min_values)
    # print("Maximum value - ", max_values)

    return min_values, max_values


def calculate_avg_min_max_values(i, forecast_arrays):
    pollutants = ['o3', 'no2', 'so2', 'co', 'pm1', 'pm25', 'pm10', 'nh3']
    avg_min_values = {}
    avg_max_values = {}

    for pollutant in pollutants:
        min_key = f'{pollutant}_Min'
        max_key = f'{pollutant}_Max'
        
        min_value = forecast_arrays[min_key][i][3]
        max_value = forecast_arrays[max_key][i][3]
        
        avg_min_values[f'{pollutant}_AVG_MIN'] = min(min_value, max_value)
        avg_max_values[f'{pollutant}_AVG_MAX'] = max(min_value, max_value)

    # print("Avg Minimum value - ", avg_min_values)
    # print("Avg Maximum value - ", avg_max_values)

    return avg_min_values, avg_max_values



if __name__=='__main__':

    # Example usage
    file_path = "Kolkata_AQI_Data.xlsx"
    # n_days = 5
    n_days = int(input("Enter the days - "))
    forecast_arrays = forecast_air_quality(file_path, n_days)

    minimum_aqi = []
    maximum_aqi = []
    avg_min_aqi = []
    avg_max_aqi = []
    
    ## Date Split
    date_range = []
    for i in range(len(forecast_arrays['o3_Min'])):
        date_range.append(forecast_arrays['o3_Min'][i][0])

    # Next n days min & max calculation
    for i in range(len(forecast_arrays['co_Max'])):
        min_values, max_values = calculate_min_max_values(i, forecast_arrays)
        avg_min_values, avg_max_values = calculate_avg_min_max_values(i, forecast_arrays)

        # For Minimum value - SUBINDEX CALCULATION , for next one day
        pm10_value = min_values['pm10_MIN']
        pm10_result = calculate_pm10_category(pm10_value)
        pm25_value = min_values['pm25_MIN']
        pm25_result = calculate_pm25_category(pm25_value)
        so2_value = min_values['so2_MIN']
        so2_result = calculate_so2_category(so2_value)
        no2_value = min_values['no2_MIN']
        no2_result = calculate_no2_category(no2_value)
        co_value = min_values['co_MIN']
        co_result = calculate_co_category(co_value)
        o3_value = min_values['o3_MIN']
        o3_result = calculate_o3_category(o3_value)
        nh3_value = min_values['nh3_MIN']
        nh3_result = calculate_nh3_category(nh3_value)

        a = aqi_calculator(pm10_result, pm25_result, so2_result,no2_result,co_result,o3_result,nh3_result)
        print("MIN AQI Value is - ",round(a))
        minimum_aqi.append(round(a))

        # For Maximum value - SUBINDEX CALCULATION, for next one day
        pm10_value = max_values['pm10_MAX']
        pm10_result = calculate_pm10_category(pm10_value)
        pm25_value = max_values['pm25_MAX']
        pm25_result = calculate_pm25_category(pm25_value)
        so2_value = max_values['so2_MAX']
        so2_result = calculate_so2_category(so2_value)
        no2_value = max_values['no2_MAX']
        no2_result = calculate_no2_category(no2_value)
        co_value = max_values['co_MAX']
        co_result = calculate_co_category(co_value)
        o3_value = max_values['o3_MAX']
        o3_result = calculate_o3_category(o3_value)
        nh3_value = max_values['nh3_MAX']
        nh3_result = calculate_nh3_category(nh3_value)

        a = aqi_calculator(pm10_result, pm25_result, so2_result,no2_result,co_result,o3_result,nh3_result)
        print("MAX AQI Value is - ",round(a))
        maximum_aqi.append(round(a))

        ###### For AVG Cases
        # For Minimum value - SUBINDEX CALCULATION , for next one day
        pm10_value = avg_min_values['pm10_AVG_MIN']
        pm10_result = calculate_pm10_category(pm10_value)
        pm25_value = avg_min_values['pm25_AVG_MIN']
        pm25_result = calculate_pm25_category(pm25_value)
        so2_value = avg_min_values['so2_AVG_MIN']
        so2_result = calculate_so2_category(so2_value)
        no2_value = avg_min_values['no2_AVG_MIN']
        no2_result = calculate_no2_category(no2_value)
        co_value = avg_min_values['co_AVG_MIN']
        co_result = calculate_co_category(co_value)
        o3_value = avg_min_values['o3_AVG_MIN']
        o3_result = calculate_o3_category(o3_value)
        nh3_value = avg_min_values['nh3_AVG_MIN']
        nh3_result = calculate_nh3_category(nh3_value)

        a = aqi_calculator(pm10_result, pm25_result, so2_result,no2_result,co_result,o3_result,nh3_result)
        print("AVG MIN AQI Value is - ",round(a))
        avg_min_aqi.append(round(a))

        # For Average Maximum value - SUBINDEX CALCULATION, for next one day
        pm10_value = avg_max_values['pm10_AVG_MAX']
        pm10_result = calculate_pm10_category(pm10_value)
        pm25_value = avg_max_values['pm25_AVG_MAX']
        pm25_result = calculate_pm25_category(pm25_value)
        so2_value = avg_max_values['so2_AVG_MAX']
        so2_result = calculate_so2_category(so2_value)
        no2_value = avg_max_values['no2_AVG_MAX']
        no2_result = calculate_no2_category(no2_value)
        co_value = avg_max_values['co_AVG_MAX']
        co_result = calculate_co_category(co_value)
        o3_value = avg_max_values['o3_AVG_MAX']
        o3_result = calculate_o3_category(o3_value)
        nh3_value = avg_max_values['nh3_AVG_MAX']
        nh3_result = calculate_nh3_category(nh3_value)

        a = aqi_calculator(pm10_result, pm25_result, so2_result,no2_result,co_result,o3_result,nh3_result)
        print("AVG MAX AQI Value is - ",round(a))
        avg_max_aqi.append(round(a))

    # Create a DataFrame
    data = {
        'Date': date_range,
        'Minimum AQI': minimum_aqi,
        'Maximum AQI': maximum_aqi,
        'Average Minimum AQI': avg_min_aqi,
        'Average Maximum AQI': avg_max_aqi
    }

    df = pd.DataFrame(data)

    # Display the DataFrame
    # df.to_excel('Result.xlsx', index=True)
    print(df)
