import requests
from . import binanceConfig
import pandas as pd


BASE_URL = binanceConfig.config['base_url']
endpoint = binanceConfig.config['endpoint']


def get_historical_data(symbol, interval):
    url = BASE_URL + endpoint
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': 100000 
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:

        data = response.json()
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert timestamp to datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Convert string values to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        return df
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

