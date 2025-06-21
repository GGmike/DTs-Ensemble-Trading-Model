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
from config import config
from subFunction.backtest import backtesting
from subFunction.calculation import ck_dup_list, cal_nCr, cal_fact
from subFunction.indicators import ichimoku_cloud, vwap, rsi, macd
from subFunction.uniqueDT import uniqueDT

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

  base_learners, used_columns = uniqueDT(cluster1_column,cal_nCr(len(cluster1_column),num_of_nodes) , num_of_nodes, X_train,y_train)

##old approach
  # if len(used_columns) == 0:

  #   base_learners = []
  #   used_columns = []
  #   dup = False

  #   loop = 500
  #   # print(f"Total distinct decision trees: {loop}")
  #   success = 0


  #   sample_df = cluster1_df


  #   while success < loop:
  #     dup = False
  #     training_columns = random.sample(list(sample_df.columns), num_of_nodes)

  #     for used_col in used_columns:
  #       if ck_dup_list(training_columns,used_col) == True:
  #         dup = True
  #         break
  #     if dup == False:
  #       success += 1
  #       X_sample = X_train[training_columns]
  #       y_sample = y_train


  #       dt = DecisionTreeClassifier(max_depth=config['max_depth'])
  #       dt.fit(X_sample, y_sample)
  #       #print('no dup: ')
  #       base_learners.append(dt)
  #       used_columns.append(training_columns)
  #       # print(success)

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


