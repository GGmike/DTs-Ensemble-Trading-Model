from subFunction.main import main
from config import config
from BinanceDataSrc.binanceRESTAPi import get_historical_data
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process config parameter.')

    parser.add_argument('--symbol', '-s', default=config['symbol'], help='Stock symbol (default: TSLA)')
    parser.add_argument('--period', '-p', default=config['period'], help='Data period (default: 1D)')
    parser.add_argument('--limit', '-l', default=1000000, help='Data limit (default: 1000000)')
    parser.add_argument('--threshold', '-t', default=config['threshold'], help='Congress Voting Threshold (default: 0.6)')
    parser.add_argument('--n_estimator', '-n', default=config['rf_n_est'], help='Random Forest Number of Estimators (default: 10000)')
    
    args = parser.parse_args()
    config['symbol'] = args.symbol
    config['limit'] = int(args.limit)
    config['threshold'] = float(args.threshold)
    config['rf_n_est'] = int(args.n_estimator)
    config['period'] = args.period

    print(f"Running with config: {config}")

    
    main()

    # print("Testing new Binance Data Source")
    # get_historical_data('SOLUSDT','5m')
    # if df is not None:
    #     print(df.head())
    #     print(df.tail())
    #     print(f"Dataframe shape: {df.shape}")
    # else:   
    #     print("Failed to retrieve data.")