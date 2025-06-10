# Stock Trading Strategy Backtester

This project implements a machine learning-based backtesting framework for stock trading strategies using technical indicators and ensemble decision trees. The code leverages historical market data to build, evaluate, and visualize trading strategies, comparing model-based signals with a simple buy-and-hold approach.

## Features

- **Technical Indicator Computation:** 
  - Simple/Exponential Moving Averages (SMA/EMA)
  - VWAP (Volume Weighted Average Price)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Ichimoku Cloud

- **Model Training and Ensemble Voting:**
  - Decision Tree base learners
  - Random feature selection
  - Ensemble voting for robust predictions

- **Backtesting Engine:**
  - Simulates trading based on generated buy/sell signals
  - Calculates key metrics (returns, drawdown, win rate, etc.)
  - Compares with buy-and-hold performance

- **Visualization:**
  - Matplotlib and Plotly charts for capital curve and buy/sell events

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- plotly
- yfinance

Install dependencies with:

```bash
pip install pandas numpy scikit-learn matplotlib plotly yfinance
```

## Usage

1. **Configuration:**  
   Edit the `config` dictionary in the script to change the symbol (default: `TSLA`), date range, and indicator parameters.

2. **Run the Script:**  
   ```bash
   python <your_script_name>.py
   ```
   Replace `<your_script_name>.py` with the filename containing the provided code.

3. **Output:**  
   - Prints evaluation metrics for the strategy and buy-and-hold.
   - Shows capital evolution plots and trade events.

## Code Structure

- **Technical Indicator Functions:**  
  Functions to compute SMA, EMA, VWAP, RSI, MACD, and Ichimoku Cloud components.

- **Backtesting:**  
  The `backtesting` function simulates trading given buy signals and computes performance statistics.

- **Model Building & Prediction:**  
  - Prepares features from technical indicators.
  - Trains an ensemble of decision trees on random subsets of features.
  - Aggregates predictions with a voting threshold.

- **Visualization:**  
  - Matplotlib for capital curve comparison.
  - Plotly for interactive price and trade event charts.

## Example

The default configuration fetches 2 years of daily TSLA data, computes various indicators, generates buy/sell signals using a voting ensemble of decision trees, and evaluates performance.

```
Voting Threshold: 0.35

Evaluation Metrics:
Accuracy Score: 0.62
F1 Score: 0.54
Precision Score: 0.60
Recall Score (Sensitivity (true positive rate)): 0.50
Recall Score (Specificity (true negative rate)): 0.70
Computational Time: 7.3 seconds

Trading Strategy:
17.5%
Maximum Draw Down: -8.2
Average Win %: 3.6
Maximum Win Trade: 10.2 on 2024-06-05
...
Buy and Hold:
12.7%
...
```

Capital curves and trade markers are displayed in interactive and static plots.

## Customization

- **Change Stock Symbol:**  
  Set `'symbol': "AAPL"` (or any other ticker) in the `config` dictionary.

- **Adjust Voting Threshold:**  
  Change `'threshold'` in the `config`.

- **Experiment with Technical Indicator Parameters:**  
  Tweak periods for SMA, EMA, MACD, etc., in the `config`.

## Notes

- The script is intended for research and educational purposes.  
- Real trading involves risks; historical performance does not guarantee future results.
- The model ensemble and feature selection are simple; further improvements are possible.

## License

Apache License 2.0

See [LICENSE](LICENSE) for details.

## Acknowledgements

- [yfinance](https://github.com/ranaroussi/yfinance) for price data
- scikit-learn, pandas, numpy, matplotlib, plotly for data science tools
