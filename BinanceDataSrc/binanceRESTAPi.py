import requests
from . import binanceConfig
from config import config
import pandas as pd
from datetime import datetime, timedelta


BASE_URL = binanceConfig.config['base_url']
endpoint = binanceConfig.config['endpoint']


start_date = datetime.now() - timedelta(days=500)
start_timestamp = int(start_date.timestamp() * 1000)

today_date = datetime.now()
end_timestamp = int(today_date.timestamp() * 1000)

def get_historical_data(symbol, interval):

    url = BASE_URL + endpoint
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': 1000 ,
        'startTime': start_timestamp,
        'endTime': end_timestamp
    }
    
    all_data_responses = []

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
        
        # if len(data) < params['limit']:
        #     break

        last_open_time = data[-1][0]
        params['startTime'] = last_open_time + 1

        if last_open_time >= params['endTime']:
            break

        # if len(all_data_responses) >= config['limit']:
        #     print(f"Warning: Reached {config['limit']} records, stopping to avoid excessive data.")
        #     break
        
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
        return df
    else:
        print("No data retrieved.")
        return None
            
