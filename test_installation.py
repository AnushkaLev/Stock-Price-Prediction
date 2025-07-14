#!/usr/bin/env python3
"""
Test Installation Script
========================

This script tests if all required packages are installed correctly
and the basic functionality works.
"""

def test_imports():
    """Test if all required packages can be imported."""
    print("🔍 Testing package imports...")
    
    try:
        import yfinance as yf
        print("✅ yfinance imported successfully")
    except ImportError as e:
        print(f"❌ yfinance import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.preprocessing import StandardScaler
        print("✅ scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ scikit-learn import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✅ seaborn imported successfully")
    except ImportError as e:
        print(f"❌ seaborn import failed: {e}")
        return False
    
    try:
        import ta
        print("✅ ta (technical analysis) imported successfully")
    except ImportError as e:
        print(f"❌ ta import failed: {e}")
        return False
    
    return True

def test_data_download():
    """Test if we can download stock data."""
    print("\n📊 Testing data download...")
    
    try:
        import yfinance as yf
        
        # Try to download a small amount of data
        ticker = yf.Ticker('AAPL')
        data = ticker.history(period='5d')
        
        if not data.empty:
            print(f"✅ Successfully downloaded {len(data)} days of AAPL data")
            print(f"   Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
            return True
        else:
            print("❌ No data downloaded")
            return False
            
    except Exception as e:
        print(f"❌ Data download failed: {e}")
        return False

def test_model_import():
    """Test if our model class can be imported."""
    print("\n🤖 Testing model import...")
    
    try:
        from stock_prediction import StockPricePredictor
        print("✅ StockPricePredictor imported successfully")
        return True
    except ImportError as e:
        print(f"❌ StockPricePredictor import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality with minimal data."""
    print("\n⚡ Testing basic functionality...")
    
    try:
        from stock_prediction import StockPricePredictor
        
        # Create predictor with very recent data
        predictor = StockPricePredictor(
            symbol='AAPL',
            start_date='2024-01-01',  # Very recent data
            end_date=None
        )
        
        # Test data download
        if predictor.download_data():
            print("✅ Data download works")
            
            # Test feature engineering
            if predictor.engineer_features():
                print("✅ Feature engineering works")
                
                # Test data preparation
                if predictor.prepare_data(test_size=0.3):
                    print("✅ Data preparation works")
                    
                    # Test model training
                    if predictor.train_model():
                        print("✅ Model training works")
                        
                        # Test evaluation
                        metrics = predictor.evaluate_model()
                        if metrics:
                            print("✅ Model evaluation works")
                            print(f"   Test R²: {metrics['test_r2']:.4f}")
                            return True
        
        print("❌ Basic functionality test failed")
        return False
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 STOCK PREDICTION INSTALLATION TEST")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test 1: Package imports
    if not test_imports():
        all_tests_passed = False
    
    # Test 2: Data download
    if not test_data_download():
        all_tests_passed = False
    
    # Test 3: Model import
    if not test_model_import():
        all_tests_passed = False
    
    # Test 4: Basic functionality
    if not test_basic_functionality():
        all_tests_passed = False
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your installation is ready to use.")
        print("\nYou can now run:")
        print("   python stock_prediction.py")
        print("   python example_usage.py")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the error messages above and install missing packages.")
        print("\nTry running:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main() 