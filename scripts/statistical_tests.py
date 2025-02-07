"""
Statistical tests for Home Deco Business Analysis
This script contains the key statistical analyses for our business metrics
"""

import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

def load_data():
    """Load and prepare datasets"""
    daily_clients = pd.read_excel('../data/daily_clients.xlsx')
    monthly_sales = pd.read_excel('../data/monthly_sales_ads_data.xlsx')
    daily_web = pd.read_excel('../data/daily_web_traffic.xlsx')
    return daily_clients, monthly_sales, daily_web

def correlation_analysis(monthly_sales):
    """Analyze correlations between key metrics"""
    metrics = [
        'subtotal_usd', 'n_items', 'unique_clients',
        'clics', 'impresiones', 'costo_ads_usd', 'visitas_web'
    ]
    
    corr_matrix = monthly_sales[metrics].corr()
    return corr_matrix

def test_weekday_effect(daily_web):
    """Test if there's a significant difference in traffic between weekdays and weekends"""
    daily_web['Fecha'] = pd.to_datetime(daily_web['Fecha'])
    daily_web['is_weekend'] = daily_web['Fecha'].dt.dayofweek.isin([5, 6])
    
    weekday_traffic = daily_web[~daily_web['is_weekend']]['visitas_web']
    weekend_traffic = daily_web[daily_web['is_weekend']]['visitas_web']
    
    t_stat, p_value = stats.ttest_ind(weekday_traffic, weekend_traffic)
    return {
        'weekday_mean': weekday_traffic.mean(),
        'weekend_mean': weekend_traffic.mean(),
        't_statistic': t_stat,
        'p_value': p_value
    }

def test_ad_spend_effectiveness(monthly_sales):
    """Test correlation between ad spend and sales"""
    correlation, p_value = stats.pearsonr(
        monthly_sales['costo_ads_usd'],
        monthly_sales['subtotal_usd']
    )
    return {
        'correlation': correlation,
        'p_value': p_value
    }

def analyze_customer_segments(daily_clients):
    """Analyze if there's significant difference in purchase behavior across segments"""
    # Calculate customer lifetime value
    customer_value = daily_clients.groupby('cliente_id')['subtotal_usd'].sum()
    
    # Create segments
    segments = pd.qcut(customer_value, q=4, labels=['Low', 'Medium', 'High', 'Premium'])
    
    # Test if segments have significantly different means
    f_stat, p_value = stats.f_oneway(
        *[customer_value[segments == segment] for segment in segments.unique()]
    )
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'segment_means': customer_value.groupby(segments).mean()
    }

def test_seasonal_effects(monthly_sales):
    """Test for significant seasonal patterns in sales"""
    monthly_sales['month'] = pd.to_datetime(monthly_sales['dateMonth']).dt.month
    
    # Compare Q4 vs rest of year
    q4_sales = monthly_sales[monthly_sales['month'].isin([10, 11, 12])]['subtotal_usd']
    other_sales = monthly_sales[~monthly_sales['month'].isin([10, 11, 12])]['subtotal_usd']
    
    t_stat, p_value = stats.ttest_ind(q4_sales, other_sales)
    return {
        'q4_mean': q4_sales.mean(),
        'other_mean': other_sales.mean(),
        't_statistic': t_stat,
        'p_value': p_value
    }

if __name__ == "__main__":
    # Load data
    daily_clients, monthly_sales, daily_web = load_data()
    
    # Run all tests
    results = {
        'correlations': correlation_analysis(monthly_sales),
        'weekday_effect': test_weekday_effect(daily_web),
        'ad_effectiveness': test_ad_spend_effectiveness(monthly_sales),
        'customer_segments': analyze_customer_segments(daily_clients),
        'seasonal_effects': test_seasonal_effects(monthly_sales)
    }
    
    # Print results
    for test_name, result in results.items():
        print(f"\n{test_name.upper()}:")
        print(result)
