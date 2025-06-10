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
  df['Close'] = df[df.columns[0]]
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


  

def main():
  #Data Fetching
  dataset = yf.download(config['symbol'],config['start'],config['end'],interval = config['period'])

  dataset['Close'].reindex(dataset.index)



  dataset['Status'] = np.where(dataset['Close'] < dataset['Close'].shift(config['lag_day']),1,0)
  dataset['Status'] = dataset['Status'].fillna(0)




  dataset['EMA1'] = dataset['Close'].ewm(span = config['ema1']).mean().fillna(0)
  dataset['EMA2'] = dataset['Close'].ewm(span = config['ema2']).mean().fillna(0)
  dataset['EMA3'] = dataset['Close'].ewm(span = config['ema3']).mean().fillna(0)

  dataset['EMA1>Close'] = np.where(dataset.EMA1 > dataset.Close[config['symbol']],1,0)
  dataset['EMA2>Close'] = np.where(dataset.EMA2 > dataset.Close[config['symbol']],1,0)
  dataset['EMA3>Close'] = np.where(dataset.EMA3 > dataset.Close[config['symbol']],1,0)

  # dataset['EMA1>EMA2'] = np.where(dataset.EMA1 > dataset.EMA2 ,1,0)
  # dataset['EMA1>EMA3'] = np.where(dataset.EMA1 > dataset.EMA3 ,1,0)
  # dataset['EMA2>EMA3'] = np.where(dataset.EMA2 > dataset.EMA3,1,0)

  dataset['EMA1>EMA2>EMA3'] = np.where( (dataset.EMA1 > dataset.EMA2) & (dataset.EMA2 > dataset.EMA3) , 1, 0)




  dataset['SMA1'] = dataset['Close'].rolling(window = config['sma1']).mean().fillna(0)
  dataset['SMA2'] = dataset['Close'].rolling(window = config['sma2']).mean().fillna(0)
  dataset['SMA3'] = dataset['Close'].rolling(window = config['sma3']).mean().fillna(0)

  dataset['SMA1>Close'] = np.where(dataset.SMA1 > dataset.Close[config['symbol']],1,0)
  dataset['SMA2>Close'] = np.where(dataset.SMA2 > dataset.Close[config['symbol']],1,0)
  dataset['SMA3>Close'] = np.where(dataset.SMA3 > dataset.Close[config['symbol']],1,0)

  # dataset['SMA1>SMA2>SMA3'] = np.where(dataset.SMA1 > dataset.SMA2 & dataset.SMA2 > dataset.SMA3,1,0)
  dataset['SMA1>SMA2>SMA3'] = np.where( (dataset.SMA1 > dataset.SMA2) & (dataset.SMA2 > dataset.SMA3) , 1, 0)



  #vwap

  dataset['VWAP'] = vwap(dataset, config['vwap'])
  dataset['Vol_SMA'] = dataset['Volume'].rolling(window = config['vol_sma']).mean().fillna(0)
  dataset['Vol_SMA>Vol'] = np.where(dataset.Vol_SMA > dataset.Volume[config['symbol']], 1 , 0)

  #max vwap


  #Cloud

  dataset['Conver_line'], dataset['Base_line'], dataset['Span_a'], dataset['Span_b'], \
  dataset['leading_span_a'], dataset['leading_span_b'], dataset['lagging'] \
  = ichimoku_cloud(dataset, config['cloud_conversion'], config['cloud_base'], config['cloud_span_b'], config['cloud_lagging'])

  dataset['conver>base'] = np.where(dataset.Conver_line > dataset.Base_line, 1, 0)
  # dataset['lagging>Close'] = np.where(dataset.lagging > dataset.Close, 1, 0 )
  dataset['lagging>Close'] = np.where(dataset.lagging > dataset.Close[config['symbol']].shift(config['cloud_lagging']), 1, 0 )
  dataset['Close>spanA'] = np.where(dataset.Close[config['symbol']] > dataset.Span_a, 1, 0)
  dataset['Close>spanB'] = np.where(dataset.Close[config['symbol']] > dataset.Span_b, 1, 0)
  dataset['Cloud_A>Cloud_B'] = np.where(dataset.leading_span_a > dataset.leading_span_b, 1, 0)

  #max cloud


  #macd

  dataset['MACD'], dataset['Signal'] = macd(dataset, config['macd_1'], config['macd_2'], config['macd_3'])
  dataset['MACD>Signal'] = np.where(dataset.MACD > dataset.Signal, 1 , 0)

  #max macd
  dataset['MACD_golden'] = np.where((dataset.MACD > dataset.Signal) & (dataset.Signal > 0), 1,0)


  #RSI

  dataset['RSI_short'] = rsi(dataset, config['rsi_short'])
  dataset['RSI_long'] = rsi(dataset, config['rsi_long'])

  dataset['RSI_short>long'] = np.where(dataset.RSI_short > dataset.RSI_long, 1, 0)
  dataset['RSI_short_bull'] = np.where((dataset['RSI_short'] >= 55) & (dataset['RSI_short'] <= 90), 1, 0)
  dataset['RSI_long_bull'] = np.where((dataset['RSI_long'] >= 50) & (dataset['RSI_long'] <= 85), 1, 0)


  # dataset['Daily_SMA'] = daily_sma(config['daily_period']).reindex(dataset.index, method='ffill')

  dataset = dataset.drop(['SMA1', 'SMA2', 'SMA3', 'EMA1', 'EMA2', 'EMA3'],axis = 1)
  dataset = dataset.drop(['VWAP', 'Vol_SMA','vol_sum', 'typical_vol'],axis = 1)
  dataset = dataset.drop(['Conver_line', 'Base_line', 'Span_a', 'Span_b', 'leading_span_a', 'leading_span_b', 'lagging'],axis = 1)
  dataset = dataset.drop(['macd', 'signal', 'MACD', 'Signal'],axis = 1)
  dataset = dataset.drop(['RSI_short', 'RSI_long'],axis = 1)



  dataset = dataset.drop(['Close', 'Low', 'High','Open','Volume'], axis=1)


  dataset = dataset[52:].fillna(0)

  y = dataset['Status']
  X = dataset.drop(['Status'], axis=1)
  cluster1_column = ['EMA1>Close', 'EMA2>Close', 'EMA3>Close', 'EMA1>EMA2>EMA3',
      'SMA1>Close', 'SMA2>Close', 'SMA3>Close', 'SMA1>SMA2>SMA3',
      'Vol_SMA>Vol', 'conver>base', 'lagging>Close', 'Close>spanA',
      'Close>spanB', 'Cloud_A>Cloud_B', 'MACD>Signal', 'RSI_short>long',
      'RSI_short_bull', 'RSI_long_bull','MACD_golden']#, 'Daily_SMA']

  cluster1_df = dataset[cluster1_column]
  num_of_nodes = 2**config['max_depth'] - 1

  used_columns = []
  timer_start = time.time()

  X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,shuffle = False) #random_state= SEED)#shuffle = False)

  if len(used_columns) == 0:

    base_learners = []
    used_columns = []
    dup = False

    loop = 500
    # print(f"Total distinct decision trees: {loop}")
    success = 0


    sample_df = cluster1_df


    while success < loop:
      dup = False
      training_columns = random.sample(list(sample_df.columns), num_of_nodes)

      for used_col in used_columns:
        if ck_dup_list(training_columns,used_col) == True:
          dup = True
          break
      if dup == False:
        success += 1
        X_sample = X_train[training_columns]
        y_sample = y_train


        dt = DecisionTreeClassifier(max_depth=config['max_depth'])
        dt.fit(X_sample, y_sample)
        #print('no dup: ')
        base_learners.append(dt)
        used_columns.append(training_columns)
        # print(success)

  # print('Trees Building Over')
  # print('Running Prediction')
  base_learner_preds = [dt.predict(X_test[col]) for dt , col in zip(base_learners, used_columns)]

  stacked_preds = np.stack(base_learner_preds, axis=-1)




  # print('Trees Building Over')
  # print('Running Prediction')
  print(f"\nVoting Threshold: {config['threshold']}")
  # base_learner_preds = [dt.predict(X_train[col]) for dt , col in zip(base_learners, used_columns)]

  # stacked_preds = np.stack(base_learner_preds, axis=-1)



  voting_one = np.mean(stacked_preds == 1, axis=1)

  final_predictions = (voting_one >= config['threshold']).astype(int)


  print('\nEvaluation Metrics:')
  print(f"Accuracy Score: {accuracy_score(y_test,final_predictions)}")
  print(f"F1 Score: {f1_score(y_test,final_predictions)}")
  print(f"Precision Score: {precision_score(y_test,final_predictions)}")
  print(f"Recall Score (Sensitivity (true positive rate)): {recall_score(y_test,final_predictions)}")
  print(f"Recall Score (Specificity (true negative rate)): {recall_score(y_test,final_predictions,pos_label = 0)}")

  timer_end = time.time()
  computational_time = timer_end - timer_start
  print(f"Computational Time: {computational_time}")

  df = pd.DataFrame(final_predictions, columns=['buy'])
  BT_df = dataset.tail(len(df))
  df = df.set_index(BT_df.index)
  temp = pd.concat([BT_df, df],axis=1 ).reindex(BT_df.index)



  test_df = yf.download(config['symbol'],config['start'],config['end'],interval = config['period'])
  BT_df = pd.concat([test_df, temp],axis = 1).reindex(temp.index)
  print('Trading Strategy:')
  final_return, max_draw_down, avg_win, avg_loss, win_rate,cap_list,buy_date,sell_date = backtesting(BT_df, stop = config['active_stop_loss'])
  #buy and hold as threshold
  BT_df['buy'] = 1
  print('Buy and Hold:')
  thre_return,thre_max_draw_down, thre_avg_win, thre_avg_loss,thre_win,thre_cap_list,thre_buy_date,thre_sell_date = backtesting(BT_df)

  plt.figure(figsize=(10, 6))
  plt.plot(BT_df.index, cap_list, label='Captial')
  plt.plot(BT_df.index, thre_cap_list, color = 'r',label='Buy and Hold')

  plt.xlabel('Time')
  plt.ylabel('Price')
  plt.title(config['symbol'])
  plt.legend()

  plt.show()


  fig = go.Figure()

  fig.add_trace(
      go.Scatter(
          x=BT_df.index,
          y=BT_df['Close'],
          mode='lines',
          name='Price')
      )

  buy_price = BT_df.loc[buy_date, 'Close']


  fig.add_trace(
      go.Scatter(
          x=buy_date,
          y=buy_price,
          mode='markers',
          marker=dict(color='green', size=10),
          name='Buy')
      )

  sell_price = BT_df.loc[sell_date, 'Close']


  fig.add_trace(
      go.Scatter(
          x=sell_date,
          y=sell_price,
          mode='markers',
          marker=dict(color='red', size=10),
          name='Sell')
      )

  buy_price = BT_df.loc[buy_date, 'Close']


  fig.add_trace(
      go.Scatter(
          x=buy_date,
          y=buy_price,
          mode='markers',
          marker=dict(color='green', size=10),
          name='Buy')
      )


  fig.update_layout(
      title=config['symbol'],
      xaxis_title='Time',
      yaxis_title='Price',
      showlegend=True)

  fig.show()


if __name__ == "__main__":
  main()
  # print('Done')
  # print('Total Time Taken: ', time.time() - start_time)
