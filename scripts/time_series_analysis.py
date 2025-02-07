"""
Time Series Analysis for Home Deco Business
This script analyzes temporal patterns and creates forecasts for key metrics
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data():
    """Load and prepare time series data"""
    # Load data
    monthly_sales = pd.read_excel('data/monthly_sales_ads_data.xlsx')
    daily_web = pd.read_excel('data/daily_web_traffic.xlsx')
    
    # Prepare monthly sales time series
    monthly_sales['dateMonth'] = pd.to_datetime(monthly_sales['dateMonth'])
    monthly_sales.set_index('dateMonth', inplace=True)
    monthly_sales.sort_index(inplace=True)
    
    # Prepare daily web traffic time series
    daily_web['Fecha'] = pd.to_datetime(daily_web['Fecha'])
    daily_web.set_index('Fecha', inplace=True)
    daily_web.sort_index(inplace=True)
    
    return monthly_sales, daily_web

def analyze_seasonality(series, period):
    """Decompose time series into trend, seasonal, and residual components"""
    decomposition = seasonal_decompose(series, period=period)
    
    # Calculate seasonal strength
    seasonal_strength = 1 - np.var(decomposition.resid) / np.var(decomposition.seasonal + decomposition.resid)
    
    return {
        'decomposition': decomposition,
        'seasonal_strength': seasonal_strength
    }

def test_stationarity(series):
    """Perform Augmented Dickey-Fuller test for stationarity"""
    result = adfuller(series.dropna())
    return {
        'test_statistic': result[0],
        'p_value': result[1],
        'is_stationary': result[1] < 0.05
    }

def create_forecast(series, periods=6):
    """Create forecast using Holt-Winters method"""
    # Split data into train and test
    train_size = int(len(series) * 0.8)
    train = series[:train_size]
    test = series[train_size:]
    
    # Fit model
    model = ExponentialSmoothing(
        train,
        seasonal_periods=12,
        trend='add',
        seasonal='add'
    ).fit()
    
    # Make predictions
    predictions = model.forecast(len(test))
    forecast = model.forecast(periods)
    
    # Calculate error
    mape = mean_absolute_percentage_error(test, predictions)
    
    return {
        'train': train,
        'test': test,
        'predictions': predictions,
        'forecast': forecast,
        'mape': mape
    }

def main():
    # Load data
    monthly_sales, daily_web = load_and_prepare_data()
    
    # Analyze sales seasonality
    sales_seasonality = analyze_seasonality(monthly_sales['subtotal_usd'], period=12)
    print("\nSales Seasonality Analysis:")
    print(f"Seasonal Strength: {sales_seasonality['seasonal_strength']:.2f}")
    
    # Test sales stationarity
    sales_stationarity = test_stationarity(monthly_sales['subtotal_usd'])
    print("\nSales Stationarity Test:")
    print(f"P-value: {sales_stationarity['p_value']:.4f}")
    print(f"Is Stationary: {sales_stationarity['is_stationary']}")
    
    # Create sales forecast
    sales_forecast = create_forecast(monthly_sales['subtotal_usd'])
    print("\nSales Forecast Performance:")
    print(f"MAPE: {sales_forecast['mape']:.2%}")
    print("\nNext 6 Months Forecast:")
    print(sales_forecast['forecast'])
    
    # Analyze web traffic seasonality
    traffic_seasonality = analyze_seasonality(daily_web['visitas_web'], period=7)
    print("\nWeb Traffic Seasonality Analysis:")
    print(f"Seasonal Strength: {traffic_seasonality['seasonal_strength']:.2f}")
    
    # Create visualizations
    plt.style.use('seaborn')
    
    # Plot sales decomposition
    sales_seasonality['decomposition'].plot()
    plt.tight_layout()
    plt.savefig('outputs/figures/sales_decomposition.png')
    plt.close()
    
    # Plot sales forecast
    plt.figure(figsize=(12, 6))
    plt.plot(sales_forecast['train'].index, sales_forecast['train'], label='Training Data')
    plt.plot(sales_forecast['test'].index, sales_forecast['test'], label='Test Data')
    plt.plot(sales_forecast['predictions'].index, sales_forecast['predictions'], label='Predictions')
    plt.plot(sales_forecast['forecast'].index, sales_forecast['forecast'], label='Forecast', linestyle='--')
    plt.legend()
    plt.title('Sales Forecast')
    plt.tight_layout()
    plt.savefig('outputs/figures/sales_forecast.png')
    plt.close()

if __name__ == "__main__":
    main()
