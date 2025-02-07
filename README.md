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
  - 50% of customers repurchase within 83 days
  - Average customer lifetime value: $78.45
  - Peak engagement: Tuesday evenings, Wednesday mornings
* **Segmentation Analysis**: Identified four distinct customer segments with Premium segment driving 45% of revenue

### Digital Performance
* **Traffic Analysis**:
  - Weekday average: 87 visits/day
  - Weekend average: 70 visits/day
  - Statistical significance: p < 0.001
* **Conversion Funnel**:
  - Impressions → Clicks: 4.03%
  - Clicks → Quotes: 5.9%
  - Quotes → Sales: 22.5%

## Statistical Analysis
* **Time Series Decomposition**: Moderate seasonality (strength = 0.51)
* **Stationarity**: Confirmed with p = 0.0002
* **Correlation Analysis**: Marketing spend shows 0.22 correlation with sales
* **Forecasting**: Achieved <15% MAPE in 6-month predictions

## Business Impact

### Optimizations Identified
1. **Marketing Timing**: Aligned ad spend with 2-month customer journey
2. **Customer Retention**: Targeted interventions at 83-day mark
3. **Traffic Quality**: Improved visitor-to-quote ratio by 31%
4. **Cost Efficiency**: Reduced customer acquisition costs by 41%

### Key Achievements
* Identified $15,000+ monthly revenue optimization potential
* Developed predictive model with 85% accuracy
* Created automated reporting dashboard for key metrics
* Established data-driven KPIs for marketing team

## Tools & Notebooks
1. `01_eda.ipynb`: Exploratory data analysis
2. `02_statistical_tests.ipynb`: Statistical validation
3. `03_time_series.ipynb`: Forecasting models
4. `/scripts`: Helper functions and utilities
5. `/outputs`: Visualizations and reports

## Future Recommendations
1. Implement machine learning for customer segmentation
2. Develop real-time conversion tracking
3. Create A/B testing framework for marketing campaigns
4. Build automated anomaly detection system
