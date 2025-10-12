
import datetime

config = {
    'limit': 1000000,
    'max_depth': 2,
    'rf_depth': 3,
    'rf_n_est': 10000,
    'symbol' : "TSLA",
    'start' : datetime.datetime.today().replace(hour=0,minute=0,second =0, microsecond =0)- datetime.timedelta(days=59),
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
    'threshold': 0.6,
    'lag_day' : -1,
    'active_stop_loss': False,
      'stop_loss' : -0.05,

}
