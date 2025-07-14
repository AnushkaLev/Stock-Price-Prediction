stock price prediction w/ linear regression + feature engineering; exploring limitations of linear models on non-linear market behavior.

- uses historical stock data to predict the next day's closing price using linear regression
- trying time series analysis
- historical data using `yfinance` from Yahoo Finance

engineered features including
- Lag Features**: Previous day's prices (1, 2, 3, 5 days ago)
- Moving Averages**: 5-day, 10-day, 20-day rolling averages
- Volatility Metrics: High-low differences and ratios
- Momentum Indicators**: Price change percentages over different periods
- Volume Analysis**: Volume moving averages and ratios
- Technical Indicators**: RSI, MACD, Bollinger Bands
- Time Features**: Day of week, month, quarter
- Price Position**: Relative position to moving averages

implemented via scikit-learn, StandardScaler, matplotlib, seaborn, Time Series Split** (80% train, 20% test); evaluation w/ MAE, MSE, R², MAPE

visualization: 
- Actual vs Predicted price comparison
- Feature importance (coefficient analysis)
- Residuals plot for model diagnostics
- Price distribution comparison


ignore, ignore
--------------------------------------------
```bash
pip install -r requirements.txt
```

(runs analysis)
```bash
python stock_prediction.py
```
