import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def create_future_dates(last_date, days=30):
    future_dates = []
    current_date = last_date
    for _ in range(days):
        current_date = current_date + timedelta(days=1)
        future_dates.append(current_date)
    return pd.Series(future_dates)

def load_and_analyze_data():
    # Load the data
    print("Loading data from historical_data.csv...")
    df = pd.read_csv('historical_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Basic statistics
    print("\nBasic Statistics:")
    print("-----------------")
    stats = df.describe()
    print(stats)
    
    # Calculate daily returns and volatility
    df['eth_returns'] = df['eth_price'].pct_change()
    df['hybrid_returns'] = df['hybrid_index'].pct_change()
    df['eth_volatility'] = df['eth_returns'].rolling(window=30).std() * (252 ** 0.5)
    df['hybrid_volatility'] = df['hybrid_returns'].rolling(window=30).std() * (252 ** 0.5)
    
    # Prepare data for predictions
    X = np.arange(len(df)).reshape(-1, 1)
    
    # Fit linear regression for ETH price
    model_eth = LinearRegression()
    model_eth.fit(X, df['eth_price'])
    
    # Create future dates for projection
    future_dates = create_future_dates(df['timestamp'].iloc[-1], days=30)
    X_future = np.arange(len(df), len(df) + 30).reshape(-1, 1)
    
    # Generate predictions
    eth_pred = model_eth.predict(X_future)
    
    # Exponential smoothing for more sophisticated predictions
    exp_model_eth = ExponentialSmoothing(
        df['eth_price'],
        seasonal_periods=7,
        trend='add',
        seasonal='add'
    ).fit()
    eth_pred_exp = exp_model_eth.forecast(30).values  # Convert to numpy array
    
    # Create visualization with projections
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Price Evolution with Projections
    plt.subplot(3, 1, 1)
    plt.plot(df['timestamp'], df['eth_price'], label='ETH Price ($)', color='blue')
    plt.plot(df['timestamp'], df['hybrid_index'], label='Hybrid Index', color='green')
    plt.plot(future_dates, eth_pred, '--', label='ETH Linear Projection', color='red')
    plt.plot(future_dates, eth_pred_exp, '--', label='ETH Exp Smoothing Projection', color='purple')
    plt.title('Price Evolution with 30-Day Projections')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Volatility
    plt.subplot(3, 1, 2)
    plt.plot(df['timestamp'], df['eth_volatility']*100, label='ETH Volatility (%)')
    plt.plot(df['timestamp'], df['hybrid_volatility']*100, label='Hybrid Index Volatility (%)')
    plt.title('Historical Volatility')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Returns Distribution
    plt.subplot(3, 1, 3)
    plt.hist(df['eth_returns'].dropna(), bins=50, alpha=0.5, label='ETH Returns', density=True)
    plt.hist(df['hybrid_returns'].dropna(), bins=50, alpha=0.5, label='Hybrid Returns', density=True)
    plt.title('Returns Distribution')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('data_analysis_with_projections.png')
    print("\nAnalysis plots saved as 'data_analysis_with_projections.png'")
    
    # Print predictions
    print("\nProjections for next 30 days:")
    print("-----------------------------")
    print("ETH Price Projections:")
    for i in range(0, 30, 5):  # Print every 5th day
        print(f"Date: {future_dates.iloc[i].strftime('%Y-%m-%d')}")
        print(f"Linear Projection: ${eth_pred[i]:.2f}")
        print(f"Exp Smoothing: ${eth_pred_exp[i]:.2f}")
        print("---")
    
    # Calculate correlation and other metrics
    correlation = df['eth_price'].corr(df['hybrid_index'])
    print(f"\nCorrelation between ETH Price and Hybrid Index: {correlation:.4f}")
    
    # Print recent data summary
    print("\nLast 7 Days Summary:")
    print("-----------------")
    print(df.tail(7)[['timestamp', 'eth_price', 'hybrid_index']].to_string(index=False))

if __name__ == "__main__":
    load_and_analyze_data() 