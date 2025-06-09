import pandas as pd
import random
import math
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf
import datetime
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score


config = {
    'max_depth': 2,
    'rf_depth': 3,
    'rf_n_est': 10000,
    'symbol' : "TSLA",
    'start' : datetime.datetime.today().replace(hour=0,minute=0,second =0, microsecond =0)- datetime.timedelta(days=729),
    'end' : datetime.datetime.today().replace(hour=0,minute=0,second =0, microsecond =0),
    'period' : "1D",
    'sma1' : 3,
    'sma2' : 5,
    'sma3' : 7,
    'ema1' : 3,
    'ema2' : 5,
    'ema3' : 10,
    'vwap' : 5,
    'vol_sma': 7,
    'macd_1' : 12,
    'macd_2' : 26,
    'macd_3' : 9,
    'rsi_short': 3,
    'rsi_long': 10, #14
    'cloud_conversion': 9,
    'cloud_base': 26,
    'cloud_span_b': 52,
    'cloud_lagging': 26,
    'daily_period': 200,
    'threshold': 0.35,
    'lag_day' : -1,
    'active_stop_loss': False,
      'stop_loss' : -0.05,

}


def ck_dup_list(list_1, list_2):
  # temp1 = list_1.sort()
  # temp2 = list_2.sort()
  # return list_1 == list_2
  return set(list_1) == set(list_2)

def cal_nCr(n,r):
  temp = n - r
  return (cal_fact(n) / (cal_fact(r) * cal_fact(temp)))

def cal_fact(num):
  if num == 0 or num == 1:
    return 1
  else:
    return(num*cal_fact(num-1))


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



def backtesting(df, initial_cap = 10000, reinvest = True, stop = False ,stop_loss = -0.05):
  temp_cap = 0
  peak = 0
  max_cap = 0
  mdd = 0
  daily_cap = initial_cap
  lossing_list = []
  winning_list = []
  cash_list = []
  cap_list = []
  last_buy = 0
  bought = False
  buy_count = 0
  buy_date = []
  sell_date = []
  lossing = 0
  win_count = 0
  loss_count = 0
  amount = 0
  max_win_date = ''
  max_loss_date = ''
  just_sell = False

  cap = initial_cap

  for i in range(len(df)):

    just_sell = False


    if(bought==True and stop == True):
      # print("stop loss")
      sl_last_buy = df['Close'].iloc[last_buy]
      sl_now_price = df['Close'].iloc[i]
      lossing = (sl_now_price/sl_last_buy) - 1
      if lossing <= stop_loss:
        bought = False
        last_buy_price = df['Close'].iloc[last_buy]
        now_price = df['Close'].iloc[i]
        cap = (now_price/last_buy_price)*cap
        if(now_price > last_buy_price):
          win_count+=1
          sell_result = (now_price/last_buy_price - 1) *100
          if len(winning_list) !=0:
            if sell_result > max(winning_list):
              max_win_date = df.index[i]
          winning_list.append(sell_result)
        else:
          loss_count+=1
          sell_result = (now_price/last_buy_price - 1) *100
          if len(lossing_list) !=0:
            if sell_result < min(lossing_list):
              max_loss_date = df.index[i]
          lossing_list.append(sell_result)



        sell_date.append(df.index[i])
        cash_list.append(cap)
        just_sell = True



    if(just_sell == False):
      if(df['buy'].iloc[i] == 1 and bought == False):
        last_buy = i
        bought = True
        buy_count+=1
        buy_date.append(df.index[i])
        cash_list.append(cap)
        #amount = cap/df['Close'].iloc[i]
        #cash_list.append(cap)

      elif(df['buy'].iloc[i] == 0 and bought == True):
        bought = False
        last_buy_price = df['Close'].iloc[last_buy]
        now_price = df['Close'].iloc[i]
        cap = (now_price/last_buy_price)*cap
        if(now_price > last_buy_price):
          win_count+=1
          sell_result = (now_price/last_buy_price - 1) *100
          if len(winning_list) !=0:
            if sell_result > max(winning_list):
              max_win_date = df.index[i]
          winning_list.append(sell_result)
        else:
          loss_count+=1
          sell_result = (now_price/last_buy_price - 1) *100
          if len(lossing_list) !=0:
            if sell_result < min(lossing_list):
              max_loss_date = df.index[i]
          lossing_list.append(sell_result)



        sell_date.append(df.index[i])
        cash_list.append(cap)

      elif (i == len(df)-1 and bought == True):
        bought = False
        last_buy_price = df['Close'].iloc[last_buy]
        now_price = df['Close'].iloc[i]
        temp_cap = (now_price/last_buy_price)*cap
        if(now_price > last_buy_price):
          win_count+=1
          sell_result = (now_price/last_buy_price - 1) *100
          if len(winning_list) !=0:
            if sell_result > max(winning_list):
              max_win_date = df.index[i]
          winning_list.append(sell_result)
        else:
          loss_count+=1
          sell_result = (now_price/last_buy_price - 1) *100
          if len(lossing_list) !=0:
            if sell_result < min(lossing_list):
              max_loss_date = df.index[i]
          lossing_list.append(sell_result)


        cap = temp_cap
        sell_date.append(df.index[i])
        cash_list.append(cap)

      elif(df['buy'].iloc[i] == 1 and bought == True):
        last_buy_price = df['Close'].iloc[last_buy]
        now_price = df['Close'].iloc[i]
        temp_cap = (now_price/last_buy_price)*cap
        cash_list.append(temp_cap)

      elif(df['buy'].iloc[i] == 0 and bought == False):
        bought = False
        cap = cap
        cash_list.append(cap)

    if(cash_list[i] > peak):
      peak = cash_list[i]
      max_cap = i
    else:
      draw_down = (cash_list[i] - peak)/peak
      if(draw_down < mdd):
        mdd = draw_down



  max_draw_down = mdd * 100

  if(len(winning_list) != 0):
    avg_win = sum(winning_list)/len(winning_list)
    max_win = max(winning_list)
  else:
    avg_win = 0
    max_win = 0
  if(len(lossing_list) != 0):
    avg_loss = sum(lossing_list)/len(lossing_list)
    max_loss = min(lossing_list)
  else:
    avg_loss = 0
    max_loss = 0
  final_return =((cap/initial_cap) - 1) *100
  if buy_count>0:
    win_rate = win_count/buy_count*100
  else:
    win_rate = 0

  print(f"{final_return}%")
  print(f"Maximum Draw Down: {max_draw_down}")
  print(f"Average Win %: {avg_win}")
  print(f"Maximum Win Trade: {max_win} on {max_win_date}")
  print(f"Maximum Loss Trade: {max_loss} on {max_loss_date}")
  print(f"Average Loss %: {avg_loss}")
  print(f"Win rate: {win_rate}%")
  print(f"Total Buy count: {buy_count}")


  return final_return, max_draw_down, avg_win, avg_loss ,win_rate ,cash_list,buy_date, sell_date




