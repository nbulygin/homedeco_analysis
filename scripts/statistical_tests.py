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

def analyze_traffic_sales_lag(daily_web, monthly_sales, max_lag=3):
    """
    Analyze the correlation between web traffic and items sold with different time lags
    
    Parameters:
    -----------
    daily_web : DataFrame
        Daily web traffic data with 'Fecha' and 'visitas_web' columns
    monthly_sales : DataFrame
        Monthly sales data with 'dateMonth' and 'n_items' columns
    max_lag : int
        Maximum number of months to lag
        
    Returns:
    --------
    dict with correlation results for each lag and the best lag statistics
    """
    # Convert daily web traffic to monthly, using first day of month
    daily_web['Fecha'] = pd.to_datetime(daily_web['Fecha'])
    monthly_web = (daily_web.set_index('Fecha')
                  .resample('M')['visitas_web']
                  .sum()
                  .reset_index())
    # Convert end of month to first day of month
    monthly_web['Fecha'] = monthly_web['Fecha'].dt.to_period('M').dt.to_timestamp()
    monthly_web.columns = ['dateMonth', 'visitas_web']
    
    # Ensure monthly_sales dateMonth is datetime
    monthly_sales['dateMonth'] = pd.to_datetime(monthly_sales['dateMonth'])
    
    # Merge datasets
    merged_data = pd.merge(
        monthly_web,
        monthly_sales[['dateMonth', 'n_items']],
        on='dateMonth',
        how='inner'
    )
    
    # Calculate correlations for different lags
    lag_results = {}
    for lag in range(max_lag + 1):
        # Create lagged web traffic
        merged_data[f'traffic_lag_{lag}'] = merged_data['visitas_web'].shift(lag)
        
        # Calculate correlation
        if lag == 0:
            correlation, p_value = stats.pearsonr(
                merged_data['visitas_web'],
                merged_data['n_items']
            )
        else:
            correlation, p_value = stats.pearsonr(
                merged_data[f'traffic_lag_{lag}'].dropna(),
                merged_data['n_items'].iloc[lag:]
            )
        
        lag_results[lag] = {
            'correlation': correlation,
            'p_value': p_value
        }
    
    # Find best lag
    best_lag = max(lag_results.items(), key=lambda x: abs(x[1]['correlation']))[0]
    
    return {
        'lag_results': lag_results,
        'best_lag': best_lag,
        'best_correlation': lag_results[best_lag]['correlation'],
        'best_p_value': lag_results[best_lag]['p_value']
    }

def analyze_purchase_frequency(monthly_sales):
    """
    Analyze if there's a significant difference in purchase behavior between
    customer segments based on purchase frequency
    """
    # Calculate purchases per customer
    customer_purchases = monthly_sales.groupby('cliente_id')['n_items'].agg(['count', 'sum'])
    
    # Segment customers
    customer_purchases['frequency'] = 'one_time'
    customer_purchases.loc[customer_purchases['count'] > 1, 'frequency'] = 'repeat'
    
    # Compare purchase volumes
    one_time = customer_purchases[customer_purchases['frequency'] == 'one_time']['sum']
    repeat = customer_purchases[customer_purchases['frequency'] == 'repeat']['sum']
    
    t_stat, p_value = stats.ttest_ind(one_time, repeat)
    
    return {
        'one_time_avg': one_time.mean(),
        'repeat_avg': repeat.mean(),
        'one_time_count': len(one_time),
        'repeat_count': len(repeat),
        't_statistic': t_stat,
        'p_value': p_value
    }

def analyze_price_sensitivity(monthly_sales):
    """
    Analyze if there's a significant relationship between price points
    and conversion rates
    """
    # Calculate average price per item for each transaction
    monthly_sales['avg_price'] = monthly_sales['subtotal_usd'] / monthly_sales['n_items']
    
    # Create price segments
    price_segments = pd.qcut(monthly_sales['avg_price'], q=3, labels=['low', 'medium', 'high'])
    monthly_sales['price_segment'] = price_segments
    
    # Calculate statistics for each segment
    segment_stats = monthly_sales.groupby('price_segment').agg({
        'n_items': 'sum',
        'impresiones': 'sum',
        'avg_price': ['mean', 'min', 'max'],
        'subtotal_usd': 'sum'
    })
    
    # Calculate conversion rates
    segment_stats['conversion_rate'] = segment_stats['n_items'] / segment_stats['impresiones']
    
    # Prepare data for ANOVA test
    segments = []
    for segment in ['low', 'medium', 'high']:
        segment_data = monthly_sales[monthly_sales['price_segment'] == segment]
        # Calculate monthly conversion rates for ANOVA
        monthly_conv = segment_data.groupby(pd.to_datetime(segment_data['dateMonth']).dt.to_period('M')).agg({
            'n_items': 'sum',
            'impresiones': 'sum'
        })
        monthly_conv['conv_rate'] = monthly_conv['n_items'] / monthly_conv['impresiones']
        segments.append(monthly_conv['conv_rate'].values)
    
    # Perform ANOVA test
    f_stat, p_value = stats.f_oneway(*segments)
    
    return {
        'segment_stats': segment_stats,
        'f_statistic': f_stat,
        'p_value': p_value
    }

def analyze_marketing_channels(monthly_sales):
    """
    Compare effectiveness of different marketing channels using ANOVA
    """
    # Calculate ROI per channel (assuming we have channel data)
    monthly_sales['roi'] = (monthly_sales['subtotal_usd'] - monthly_sales['costo_ads_usd']) / monthly_sales['costo_ads_usd']
    
    # If we have channel data, group by it
    if 'channel' in monthly_sales.columns:
        channel_stats = monthly_sales.groupby('channel').agg({
            'roi': ['mean', 'std', 'count'],
            'subtotal_usd': 'sum',
            'costo_ads_usd': 'sum'
        })
        
        # ANOVA test for ROI across channels
        channels = [group['roi'].values for name, group in monthly_sales.groupby('channel')]
        f_stat, p_value = stats.f_oneway(*channels)
        
        return {
            'channel_stats': channel_stats,
            'f_statistic': f_stat,
            'p_value': p_value
        }
    else:
        return {
            'error': 'Channel data not available in dataset'
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
        'seasonal_effects': test_seasonal_effects(monthly_sales),
        'traffic_sales_lag': analyze_traffic_sales_lag(daily_web, monthly_sales),
        'purchase_frequency': analyze_purchase_frequency(monthly_sales),
        'price_sensitivity': analyze_price_sensitivity(monthly_sales),
        'marketing_channels': analyze_marketing_channels(monthly_sales)
    }
    
    # Print results
    for test_name, result in results.items():
        print(f"\n{test_name.upper()}:")
        print(result)
