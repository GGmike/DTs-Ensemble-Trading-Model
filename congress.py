from subFunction.main import main
from config import config
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process config parameter.')

    parser.add_argument('--symbol', '-s', default=config['symbol'], help='Stock symbol (default: TSLA)')
    parser.add_argument('--threshold', '-t', default=config['threshold'], help='Congress Voting Threshold (default: 0.6)')
    parser.add_argument('--n_estimator', '-n', default=config['rf_n_est'], help='Random Forest Number of Estimators (default: 10000)')
    
    args = parser.parse_args()
    config['symbol'] = args.symbol
    config['threshold'] = float(args.threshold)
    config['rf_n_est'] = int(args.n_estimator)

    print(f"Running with config: {config}")

    
    main()
