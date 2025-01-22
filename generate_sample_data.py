import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import yfinance as yf

def fetch_eth_data():
    """Fetch real ETH price data from CoinGecko"""
    url = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': '365',  # Last year of data
        'interval': 'daily'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Convert price data to DataFrame
        prices = data['prices']
        df_eth = pd.DataFrame(prices, columns=['timestamp', 'eth_price'])
        df_eth['timestamp'] = pd.to_datetime(df_eth['timestamp'], unit='ms')
        return df_eth
    except Exception as e:
        print(f"Error fetching ETH data: {e}")
        return None

def fetch_sp500_data():
    """Fetch real S&P 500 data using yfinance"""
    try:
        spy = yf.download('^GSPC', start='2024-01-01', end='2025-01-31')
        df_sp500 = spy[['Close']].reset_index()
        df_sp500.columns = ['timestamp', 'sp500_index']
        return df_sp500
    except Exception as e:
        print(f"Error fetching S&P 500 data: {e}")
        return None

# Fetch real historical data
df_eth = fetch_eth_data()
df_sp500 = fetch_sp500_data()

if df_eth is not None and df_sp500 is not None:
    # Merge and process historical data
    df_eth['timestamp'] = df_eth['timestamp'].dt.date
    df_sp500['timestamp'] = pd.to_datetime(df_sp500['timestamp']).dt.date
    
    df_historical = pd.merge(df_eth, df_sp500, on='timestamp', how='inner')
    df_historical['hybrid_index'] = (df_historical['eth_price'] / df_historical['sp500_index']) * 1000
    
    # Generate future dates (from February 2025 to December 2025)
    last_date = datetime(2025, 1, 31).date()
    end_date = datetime(2025, 12, 31).date()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), end=end_date, freq='D')
    
    # Generate predictions for future dates
    def generate_future_predictions(historical_data, future_dates):
        # Calculate trends from last 90 days of historical data
        recent_data = historical_data.tail(90)
        eth_daily_return = np.mean(np.diff(np.log(recent_data['eth_price'])))
        sp500_daily_return = np.mean(np.diff(np.log(recent_data['sp500_index'])))
        
        # Last known values
        last_eth = historical_data['eth_price'].iloc[-1]
        last_sp500 = historical_data['sp500_index'].iloc[-1]
        
        # Generate future values with increasing uncertainty
        n_days = len(future_dates)
        time_factor = np.linspace(0.001, 0.003, n_days)  # Increasing volatility
        
        # ETH predictions
        eth_trend = np.exp(eth_daily_return * np.arange(n_days))
        eth_noise = np.random.normal(0, np.std(recent_data['eth_price']) * time_factor)
        future_eth = last_eth * eth_trend * (1 + eth_noise)
        
        # S&P 500 predictions
        sp500_trend = np.exp(sp500_daily_return * np.arange(n_days))
        sp500_noise = np.random.normal(0, np.std(recent_data['sp500_index']) * time_factor)
        future_sp500 = last_sp500 * sp500_trend * (1 + sp500_noise)
        
        # Calculate future hybrid index
        future_hybrid = (future_eth / future_sp500) * 1000
        
        return pd.DataFrame({
            'timestamp': future_dates.date,
            'eth_price': future_eth,
            'sp500_index': future_sp500,
            'hybrid_index': np.nan,  # Set actual values to NaN for future dates
            'predicted_hybrid_index': future_hybrid  # Only predictions for future dates
        })
    
    # Generate future data
    df_future = generate_future_predictions(df_historical, future_dates)
    
    # For historical data, set predicted values equal to actual values
    df_historical['predicted_hybrid_index'] = df_historical['hybrid_index']
    
    # Combine historical and future data
    df_historical['timestamp'] = pd.to_datetime(df_historical['timestamp'])
    df_future['timestamp'] = pd.to_datetime(df_future['timestamp'])
    
    df_combined = pd.concat([df_historical, df_future], ignore_index=True)
    
    # Save to CSV
    df_combined.to_csv('historical_data.csv', index=False)
    print("Données générées dans 'historical_data.csv'")
    print(f"Période couverte : {df_combined['timestamp'].min().strftime('%Y-%m-%d')} à {df_combined['timestamp'].max().strftime('%Y-%m-%d')}")
    print(f"Nombre de jours : {len(df_combined)}")
    print("\nRésumé des données :")
    print(f"ETH Price range: ${df_combined['eth_price'].min():.0f} - ${df_combined['eth_price'].max():.0f}")
    print(f"S&P 500 range: {df_combined['sp500_index'].min():.0f} - {df_combined['sp500_index'].max():.0f}")
    print(f"Hybrid Index range: {df_combined['hybrid_index'].dropna().min():.0f} - {df_combined['hybrid_index'].dropna().max():.0f}")
else:
    print("Erreur lors de la récupération des données historiques") 