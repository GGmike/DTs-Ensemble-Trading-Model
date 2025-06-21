import numpy as np
import yfinance as yf
import datetime
from config import config


#Ichimoku Cloud
def ichimoku_cloud(df, con_period, base_period, span_b_period, lag_period):
  conver_line = (df['High'].rolling(window = con_period).max() + df['Low'].rolling(window = con_period).min())/2
  base_line = (df['High'].rolling(window = base_period).max() + df['Low'].rolling(window = base_period).min())/2

  span_a = ((conver_line + base_line)/2)
  span_b = ((df['High'].rolling(window = span_b_period).max() + df['Low'].rolling(window = span_b_period).min())/2)

  leading_span_a = ((conver_line + base_line)/2).shift(lag_period)
  leading_span_b = ((df['High'].rolling(window = span_b_period).max() + df['Low'].rolling(window = span_b_period).min())/2).shift(lag_period)

  lagging_line = df['Close']#.shift(-lag_period)
  return conver_line, base_line, span_a, span_b, leading_span_a, leading_span_b, lagging_line


#vwap
def vwap(df,n):
  df['vol_sum'] = df['Volume'].cumsum()
  df['typical_vol'] = (df['Close']*df['Volume']).cumsum()
  df['VWAP'] = df['typical_vol']/df['vol_sum']
  return df['VWAP']

#rsi
def rsi(df,n):
  diff = df['Close'].diff()
  gain = (diff.where(diff > 0, 0)).rolling(window = n).mean()
  loss = (-diff.where(diff < 0,0)).rolling(window = n).mean()
  rs = gain/loss
  return (100 - (100/(1 + rs)))

#macd
def macd(df,a,b,c):
  ema_1 = df['Close'].ewm(span = a).mean().fillna(0)
  ema_2 = df['Close'].ewm(span = b).mean().fillna(0)
  df['macd'] = ema_1 - ema_2
  df['signal'] = df['macd'].ewm(span = c).mean().fillna(0)
  return df['macd'],df['signal']

#SMA Threshold
def daily_sma(a):
  daily_start = config['start'] - datetime.timedelta(days=a)
  temp_df = yf.download(config['symbol'],daily_start,config['end'])
  temp_df['SMA'] = temp_df['Close'].rolling(window = a).mean()
  temp_df['daily_SMA>Close'] = np.where(temp_df.SMA > temp_df.Close, 1,0)
  temp_df.index = temp_df.index.tz_localize('UTC') if temp_df.index.tz is None else temp_df.index.tz_convert('UTC')

  return temp_df['daily_SMA>Close'][a:].shift(1)






  