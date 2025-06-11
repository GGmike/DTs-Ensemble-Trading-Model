


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
