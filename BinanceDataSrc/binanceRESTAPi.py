import requests
from . import binanceConfig
from config import config
import pandas as pd
from datetime import datetime, timedelta


BASE_URL = binanceConfig.config['base_url']
endpoint = binanceConfig.config['endpoint']


def get_historical_data(symbol, interval):

    url = BASE_URL + endpoint
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': 1000 
    }
    
    all_data_responses = []
    current_start = None

    if params.get('endTime') is None:
        params['endTime'] = int(datetime.now().timestamp() * 1000)

    while True:

        response = requests.get(url, params=params)

        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            break
            
        data = response.json()

        if not data:
            break


        all_data_responses.extend(data)
        
        if len(data) < params['limit']:
            break

        current_start = data[0][0] - 1

        params['startTime'] = current_start
        

        if current_start >= params['endTime']:
            break

        if len(all_data_responses) >= config['limit']:
            print(f"Warning: Reached {config['limit']} records, stopping to avoid excessive data.")
            break
        
        print(f"Added {len(data)} records, total so far: {len(all_data_responses)}")
        
    if all_data_responses:

        df = pd.DataFrame(all_data_responses, columns=[
            'open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        # Convert timestamp to datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Convert string values to float
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)

        print(f"Final dataframe shape: {df.shape}")
        print(df.head())
        print(df.tail())
        print(df.dtypes)
        print(df.index)
        return df
    else:
        print("No data retrieved.")
        return None
            
