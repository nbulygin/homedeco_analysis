# Home Deco E-commerce Analysis

## Project Overview
Analyzed sales and marketing performance for a home decoration e-commerce business to identify growth opportunities and optimize digital marketing ROI. The analysis covered customer behavior, marketing effectiveness, and sales patterns from 2021 to 2024.

## Skills & Tools Used
- Python (Pandas, NumPy, Statsmodels)
- Time Series Analysis
- Statistical Testing
- Data Visualization (Matplotlib, Seaborn)
- Marketing Analytics

## Key Findings

### Sales & Marketing Performance
* **Revenue Trend**: Identified 2-month lag between marketing activities and sales conversion
* **Marketing Efficiency**: 
  - ROI improved from 37.3x to 156.6x
  - CTR increased 86% (2.16% to 4.03%)
  - CPC reduced 41% ($0.41 to $0.24)

### Customer Behavior
* **Purchase Patterns**:
  - 12.9% are repeat customers, purchasing 2.79x more items
  - Average items: 2.84 (one-time) vs 7.92 (repeat) customers
  - 50% of recurring customers purchase within 90 days
  - Peak engagement: Tuesday evenings, Wednesday mornings
* **Price Sensitivity**:
  - Medium price range ($117-$138) shows highest conversion (0.168%)
  - No significant price sensitivity (p = 0.18)
* **Sales Anomalies**:
  - Significant sales spike in April 2022 (74.4% above mean)
  - Non-stationary sales pattern (p = 0.96)
  - Strong autocorrelation in daily sales (p < 0.01)

### Digital Performance
* **Traffic Analysis**:
  - Weekday average: 87 visits/day
  - Weekend average: 70 visits/day
  - Statistical significance: p < 0.001
* **Conversion Funnel**:
  - Best performing price segment: $117-$138 (medium range)
  - Strong relationship between traffic and sales (R² = 32% at 2-month lag)

## Statistical Analysis
* **Time Series Decomposition**: Moderate seasonality (strength = 0.51)
* **Customer Journey**: 2-month optimal lag between traffic and sales (p < 0.001)
* **Purchase Behavior**: Significant difference between one-time and repeat customers (p < 0.0001)
* **Forecasting**: Achieved <15% MAPE in 6-month predictions

## Business Impact

### Optimizations Identified
1. **Marketing Timing**: Align campaigns with 2-month customer journey lag
2. **Customer Retention**: Focus on converting 87.1% one-time customers
3. **Price Strategy**: Optimize medium price range ($117-$138)
4. **Traffic Quality**: Target 32% explained variance in sales

### Key Achievements
* Identified $1.1K-$15K+ monthly revenue potential by converting one-time buyers (2-10% of 87.1% non-recurring customer base) to repeat customers who purchase 2.79x more
* Built reliable sales volume forecasting model with 85% prediction accuracy over 6-month period
* Created automated reporting dashboard for key metrics
* Established data-driven KPIs for marketing team
* Discovered October peak season with 21.8% higher web traffic, enabling targeted marketing campaigns

## Tools & Notebooks
1. `01_eda.ipynb`: Exploratory data analysis
2. `02_statistical_tests.ipynb`: Statistical validation
3. `03_time_series.ipynb`: Forecasting models
4. `/outputs`: Visualizations and reports

## Future Recommendations
1. Implement machine learning for customer segmentation
2. Develop real-time conversion tracking
3. Create A/B testing framework for marketing campaigns
4. Build automated anomaly detection system
