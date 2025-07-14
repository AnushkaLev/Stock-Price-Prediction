"""
Stock Price Prediction using Linear Regression
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
import ta

class StockPricePredictor:
    """
    A class to predict stock prices using linear regression with feature engineering.
    """
    
    def __init__(self, symbol='AAPL', start_date='2022-01-01', end_date=None):
        """
        Initialize the stock predictor.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'META', 'COIN')
            start_date (str): Start date for data collection
            end_date (str): End date for data collection (defaults to today)
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def download_data(self):
        """Download historical stock data using yfinance."""
        print(f"Downloading {self.symbol} data from {self.start_date} to {self.end_date}...")
        
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(start=self.start_date, end=self.end_date)
            
            if self.data.empty:
                raise ValueError(f"No data found for {self.symbol}")
                
            print(f"Downloaded {len(self.data)} days of data")
            print(f"Price range: ${self.data['Close'].min():.2f} - ${self.data['Close'].max():.2f}")
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            return False
            
        return True
    
    def engineer_features(self):
        """
        Engineer features for the prediction model.
        This is where we create meaningful features that might help predict stock prices.
        """
        print("Engineering features...")
        
        if self.data is None:
            print("No data available. Please download data first.")
            return False
        
        # Create a copy to avoid modifying original data
        df = self.data.copy()
        
        # 1. LAG FEATURES - Previous day's prices
        df['lag_1_close'] = df['Close'].shift(1)
        df['lag_2_close'] = df['Close'].shift(2)
        df['lag_3_close'] = df['Close'].shift(3)
        df['lag_5_close'] = df['Close'].shift(5)
        
        # 2. ROLLING AVERAGES - Moving averages
        df['ma_5'] = df['Close'].rolling(window=5).mean()
        df['ma_10'] = df['Close'].rolling(window=10).mean()
        df['ma_20'] = df['Close'].rolling(window=20).mean()
        
        # 3. VOLATILITY FEATURES
        df['high_low_diff'] = df['High'] - df['Low']
        df['high_low_ratio'] = df['High'] / df['Low']
        
        # 4. MOMENTUM FEATURES
        df['price_change'] = df['Close'].pct_change()
        df['price_change_2d'] = df['Close'].pct_change(periods=2)
        df['price_change_5d'] = df['Close'].pct_change(periods=5)
        
        # 5. VOLUME FEATURES
        df['volume_ma_5'] = df['Volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma_5']
        
        # 6. TECHNICAL INDICATORS
        # RSI (Relative Strength Index)
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        
        # 7. TIME-BASED FEATURES
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # 8. PRICE POSITION FEATURES
        df['price_vs_ma5'] = df['Close'] / df['ma_5'] - 1
        df['price_vs_ma20'] = df['Close'] / df['ma_20'] - 1
        
        # Store feature names for later use
        self.feature_names = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        print(f"Created {len(self.feature_names)} features:")
        for i, feature in enumerate(self.feature_names, 1):
            print(f"   {i:2d}. {feature}")
        
        self.data = df
        return True
    
    def prepare_data(self, test_size=0.2):
        """
        Prepare data for training and testing.
        
        Args:
            test_size (float): Proportion of data to use for testing
        """
        print("Preparing data for modeling...")
        
        # Remove rows with NaN values (from lag features and rolling averages)
        df_clean = self.data.dropna()
        
        # Define features and target
        X = df_clean[self.feature_names]
        y = df_clean['Close']
        
        # Split data chronologically (time series split)
        split_idx = int(len(df_clean) * (1 - test_size))
        
        self.X_train = X[:split_idx]
        self.X_test = X[split_idx:]
        self.y_train = y[:split_idx]
        self.y_test = y[split_idx:]
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        print(f"Features shape: {self.X_train.shape[1]}")
        
        return True
    
    def train_model(self):
        """Train the linear regression model."""
        print("🤖 Training linear regression model...")
        
        self.model = LinearRegression()
        self.model.fit(self.X_train_scaled, self.y_train)
        
        # Make predictions
        self.y_train_pred = self.model.predict(self.X_train_scaled)
        self.y_test_pred = self.model.predict(self.X_test_scaled)
        
        print("Model training completed!")
        return True
    
    def evaluate_model(self):
        """Evaluate the model performance."""
        print("\nMODEL EVALUATION")
        print("=" * 50)
        
        # Calculate metrics
        train_mae = mean_absolute_error(self.y_train, self.y_train_pred)
        test_mae = mean_absolute_error(self.y_test, self.y_test_pred)
        
        train_mse = mean_squared_error(self.y_train, self.y_train_pred)
        test_mse = mean_squared_error(self.y_test, self.y_test_pred)
        
        train_r2 = r2_score(self.y_train, self.y_train_pred)
        test_r2 = r2_score(self.y_test, self.y_test_pred)
        
        # Print results
        print(f"TRAINING SET:")
        print(f"   MAE: ${train_mae:.2f}")
        print(f"   MSE: ${train_mse:.2f}")
        print(f"   R²:  {train_r2:.4f}")
        
        print(f"\nTEST SET:")
        print(f"   MAE: ${test_mae:.2f}")
        print(f"   MSE: ${test_mse:.2f}")
        print(f"   R²:  {test_r2:.4f}")
        
        # Calculate percentage errors
        train_mape = np.mean(np.abs((self.y_train - self.y_train_pred) / self.y_train)) * 100
        test_mape = np.mean(np.abs((self.y_test - self.y_test_pred) / self.y_test)) * 100
        
        print(f"\nPERCENTAGE ERRORS:")
        print(f"   Training MAPE: {train_mape:.2f}%")
        print(f"   Test MAPE: {test_mape:.2f}%")
        
        return {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mape': train_mape,
            'test_mape': test_mape
        }
    
    def plot_results(self):
        """Create comprehensive visualization of results."""
        print("\nCreating visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Stock Price Prediction Results for {self.symbol}', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted Prices (Test Set)
        test_dates = self.data.index[-len(self.y_test):]
        axes[0, 0].plot(test_dates, self.y_test, label='Actual', linewidth=2, color='blue')
        axes[0, 0].plot(test_dates, self.y_test_pred, label='Predicted', linewidth=2, color='red', linestyle='--')
        axes[0, 0].set_title('Actual vs Predicted Prices (Test Set)')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Feature Importance (Coefficients)
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        top_features = feature_importance.head(10)
        colors = ['red' if x < 0 else 'green' for x in top_features['Coefficient']]
        
        axes[0, 1].barh(range(len(top_features)), top_features['Coefficient'], color=colors, alpha=0.7)
        axes[0, 1].set_yticks(range(len(top_features)))
        axes[0, 1].set_yticklabels(top_features['Feature'])
        axes[0, 1].set_title('Top 10 Feature Coefficients')
        axes[0, 1].set_xlabel('Coefficient Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residuals Plot
        residuals = self.y_test - self.y_test_pred
        axes[1, 0].scatter(self.y_test_pred, residuals, alpha=0.6, color='purple')
        axes[1, 0].axhline(y=0, color='red', linestyle='--')
        axes[1, 0].set_title('Residuals Plot')
        axes[1, 0].set_xlabel('Predicted Price')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Price Distribution Comparison
        axes[1, 1].hist(self.y_test, bins=20, alpha=0.7, label='Actual', color='blue')
        axes[1, 1].hist(self.y_test_pred, bins=20, alpha=0.7, label='Predicted', color='red')
        axes[1, 1].set_title('Price Distribution Comparison')
        axes[1, 1].set_xlabel('Price ($)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.symbol}_prediction_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved as '{}_prediction_results.png'".format(self.symbol))
    
    def analyze_feature_correlations(self):
        """Analyze correlations between features and target."""
        print("\n🔍 FEATURE CORRELATION ANALYSIS")
        print("=" * 50)
        
        # Calculate correlations with target
        correlations = []
        for feature in self.feature_names:
            corr = self.data[feature].corr(self.data['Close'])
            correlations.append((feature, corr))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print("Top 10 features by correlation with closing price:")
        for i, (feature, corr) in enumerate(correlations[:10], 1):
            direction = "positive" if corr > 0 else "negative"
            print(f"   {i:2d}. {feature:20s}: {corr:6.3f} ({direction})")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print(f"Starting Stock Price Prediction Analysis for {self.symbol}")
        print("=" * 60)
        
        # Step 1: Download data
        if not self.download_data():
            return False
        
        # Step 2: Engineer features
        if not self.engineer_features():
            return False
        
        # Step 3: Prepare data
        if not self.prepare_data():
            return False
        
        # Step 4: Train model
        if not self.train_model():
            return False
        
        # Step 5: Evaluate model
        metrics = self.evaluate_model()
        
        # Step 6: Analyze features
        self.analyze_feature_correlations()
        
        # Step 7: Create visualizations
        self.plot_results()
        
        print("\nAnalysis completed!")
        print("=" * 60)
        
        return metrics


def main():
    """Main function to run the stock prediction analysis."""
    
    # List of popular stocks to choose from
    stocks = ['AAPL', 'META', 'COIN', 'NVDA', 'TSLA', 'MSFT', 'GOOGL']
    
    print("STOCK PRICE PREDICTION WITH LINEAR REGRESSION")
    print("=" * 60)
    print("Available stocks:", ', '.join(stocks))
    print()
    
    # You can change the stock symbol here
    symbol = 'AAPL'  # Change this to any stock you want to analyze
    
    # Create and run the predictor
    predictor = StockPricePredictor(symbol=symbol, start_date='2022-01-01')
    results = predictor.run_complete_analysis()
    
    if results:
        print(f"\n💡 KEY INSIGHTS:")
        print(f"   • Model explains {results['test_r2']:.1%} of price variance")
        print(f"   • Average prediction error: ${results['test_mae']:.2f}")
        print(f"   • Percentage error: {results['test_mape']:.1f}%")
        print(f"   • Linear regression limitations: Stock prices are non-linear!")
        print(f"   • Feature engineering helps but can't overcome market complexity")


if __name__ == "__main__":
    main() 
