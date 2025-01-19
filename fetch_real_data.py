import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv
import yfinance as yf

# Load environment variables
load_dotenv()

def fetch_eth_data():
    """
    Fetch ETH/USD historical data from CoinGecko
    """
    # CoinGecko API endpoint for ETH historical data (daily)
    url = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
    
    # Get data for the last 365 days
    params = {
        'vs_currency': 'usd',
        'days': '365',
        'interval': 'daily'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Convert price data to DataFrame
        df_eth = pd.DataFrame(data['prices'], columns=['timestamp', 'eth_price'])
        df_eth['timestamp'] = pd.to_datetime(df_eth['timestamp'], unit='ms')
        
        return df_eth
        
    except Exception as e:
        print(f"Error fetching ETH data: {str(e)}")
        return None

def fetch_sp500_data():
    """
    Fetch S&P 500 historical data from Yahoo Finance
    """
    try:
        print("Fetching S&P 500 historical data from Yahoo Finance...")
        # Get SPY (S&P 500 ETF) data for the last year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
        
        # Handle multi-index DataFrame
        if isinstance(spy.columns, pd.MultiIndex):
            spy = spy.loc[:, ('Close', 'SPY')]  # Select Close price from SPY
        else:
            spy = spy['Close']  # Fallback for single-index
        
        # Convert to DataFrame with the format we need
        df_sp500 = pd.DataFrame({
            'timestamp': spy.index,
            'sp500_index': spy.values
        })
        
        print(f"Successfully fetched {len(df_sp500)} days of S&P 500 data")
        return df_sp500
        
    except Exception as e:
        print(f"Error fetching S&P 500 data: {str(e)}")
        if 'spy' in locals():
            print("Available columns:", spy.columns)
        return None

def merge_and_process_data():
    """
    Merge ETH and S&P 500 data and calculate hybrid index
    """
    # Fetch ETH data first
    print("Fetching ETH data...")
    df_eth = fetch_eth_data()
    if df_eth is None:
        return None
    
    # Then fetch S&P 500 data
    print("Fetching S&P 500 data...")
    df_sp500 = fetch_sp500_data()
    if df_sp500 is None:
        return None
    
    # Convert timestamps to string date format for merging
    print("Processing data...")
    df_eth['date'] = df_eth['timestamp'].dt.strftime('%Y-%m-%d')
    df_sp500['date'] = df_sp500['timestamp'].dt.strftime('%Y-%m-%d')
    
    # Merge datasets on date
    df = pd.merge(df_eth, df_sp500, on='date', how='inner')
    
    # Keep only the timestamp from df_eth and drop the extra columns
    df = df.drop(['timestamp_y'], axis=1)
    df = df.rename(columns={'timestamp_x': 'timestamp'})
    df = df.drop(['date'], axis=1)
    
    # Calculate hybrid index
    df['hybrid_index'] = (df['eth_price'] / df['sp500_index']) * 1000
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    return df

def main():
    print("Fetching real historical data...")
    df = merge_and_process_data()
    
    if df is not None:
        # Save to CSV
        df.to_csv('historical_data.csv', index=False)
        print(f"Data saved to historical_data.csv")
        print(f"Number of records: {len(df)}")
        print("\nSample of the data:")
        print(df.head())
    else:
        print("Failed to fetch data")

if __name__ == "__main__":
    main() 