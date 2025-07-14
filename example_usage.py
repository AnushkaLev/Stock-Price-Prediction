#!/usr/bin/env python3
"""
Example Usage of Stock Price Prediction Model
============================================

This script demonstrates how to use the StockPricePredictor class
with different stocks and configurations.
"""

from stock_prediction import StockPricePredictor
import pandas as pd

def analyze_multiple_stocks():
    """Analyze multiple stocks and compare their performance."""
    
    stocks = ['AAPL', 'META', 'NVDA', 'TSLA']
    results = {}
    
    print("🔍 COMPARING MULTIPLE STOCKS")
    print("=" * 50)
    
    for stock in stocks:
        print(f"\n📊 Analyzing {stock}...")
        
        try:
            # Create predictor
            predictor = StockPricePredictor(
                symbol=stock,
                start_date='2023-01-01',  # More recent data
                end_date=None
            )
            
            # Run analysis
            metrics = predictor.run_complete_analysis()
            
            if metrics:
                results[stock] = metrics
                print(f"✅ {stock} analysis completed!")
                
        except Exception as e:
            print(f"❌ Error analyzing {stock}: {e}")
    
    # Compare results
    if results:
        print("\n📊 COMPARISON SUMMARY")
        print("=" * 50)
        
        comparison_data = []
        for stock, metrics in results.items():
            comparison_data.append({
                'Stock': stock,
                'Test R²': metrics['test_r2'],
                'Test MAE': metrics['test_mae'],
                'Test MAPE': metrics['test_mape']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False, float_format='%.4f'))
        
        # Find best performing stock
        best_stock = max(results.items(), key=lambda x: x[1]['test_r2'])
        print(f"\n🏆 Best performing stock: {best_stock[0]} (R² = {best_stock[1]['test_r2']:.4f})")

def custom_analysis():
    """Demonstrate custom analysis with different parameters."""
    
    print("\n🔧 CUSTOM ANALYSIS EXAMPLE")
    print("=" * 50)
    
    # Create predictor with custom parameters
    predictor = StockPricePredictor(
        symbol='COIN',  # Coinbase stock
        start_date='2022-01-01',
        end_date='2024-01-01'
    )
    
    # Step-by-step analysis
    print("1. Downloading data...")
    if predictor.download_data():
        print("2. Engineering features...")
        if predictor.engineer_features():
            print("3. Preparing data with custom split...")
            if predictor.prepare_data(test_size=0.15):  # 15% for testing
                print("4. Training model...")
                if predictor.train_model():
                    print("5. Evaluating performance...")
                    metrics = predictor.evaluate_model()
                    
                    print(f"\n📊 COIN Analysis Results:")
                    print(f"   Test R²: {metrics['test_r2']:.4f}")
                    print(f"   Test MAE: ${metrics['test_mae']:.2f}")
                    print(f"   Test MAPE: {metrics['test_mape']:.2f}%")
                    
                    # Create visualizations
                    predictor.plot_results()
                    
                    # Analyze feature correlations
                    predictor.analyze_feature_correlations()

def quick_test():
    """Quick test with a single stock."""
    
    print("\n⚡ QUICK TEST")
    print("=" * 30)
    
    # Quick analysis with MSFT
    predictor = StockPricePredictor(
        symbol='MSFT',
        start_date='2023-06-01',  # Last 6 months
        end_date=None
    )
    
    results = predictor.run_complete_analysis()
    
    if results:
        print(f"\n✅ Quick test completed!")
        print(f"   Microsoft stock prediction accuracy: {results['test_r2']:.1%}")

def main():
    """Main function to run examples."""
    
    print("📈 STOCK PREDICTION EXAMPLES")
    print("=" * 60)
    
    # Example 1: Quick test
    quick_test()
    
    # Example 2: Custom analysis
    custom_analysis()
    
    # Example 3: Multiple stocks comparison
    analyze_multiple_stocks()
    
    print("\n🎉 All examples completed!")
    print("Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main() 