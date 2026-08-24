# Script 7: Main Script - gather all modules, plug them into each other and start

# 1. Imports
import logging
import numpy as np
import matplotlib.pyplot as plt

from data_ingestion import DataIngestion
from signal_generator import SignalGenerator  
from risk_models import RiskModel
from bl_engine import BlackLittermanEngine
from portfolio_optimiser import PortfolioOptimizer 
from backtester import WalkForwardBacktester

# 2. Format Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 3. Define master function
def main():
    logging.info("Starting Black-Litterman Portfolio Optimization Pipeline...")
    
    # i) Configuration (Using recent dates to ensure yfinance fundamental data exists)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'BRK-B', 'JNJ', 'JPM', 'V', 'PG'] # modular: can be replaced with any set of tickers
    start_date = "2025-08-01"
    end_date = "2026-08-20"
    
    # ii) Instantiate all modular engines
    data_engine = DataIngestion(tickers, start_date, end_date)
    risk_engine = RiskModel(risk_aversion=2.5)
    signal_engine = SignalGenerator(expected_outperformance=0.03)
    bl_engine = BlackLittermanEngine(tau=0.05, confidence=0.65)
    optimizer_engine = PortfolioOptimizer(risk_aversion=2.5, turnover_penalty=0.001)
    backtester = WalkForwardBacktester(rebalance_freq=21, lookback_window=63)
    
    # iii) Execute Data Pipeline
    try:
        prices = data_engine.fetch_pricing_data()
        fcf_yields = data_engine.fetch_fundamental_fcf_yield(prices)
    except Exception as e:
        logging.error(f"Data pipeline failed: {e}") # defensive programming: API shield
        return

    # iv) Run the Walk-Forward Backtest
    bl_strategy_returns = backtester.run_backtest(
        prices=prices,
        fcf_yields=fcf_yields,
        risk_engine=risk_engine,
        signal_engine=signal_engine,
        bl_engine=bl_engine,
        optimizer=optimizer_engine
    )
    
    # v) Create a Benchmark (Equal-Weight Portfolio 1/N)
    daily_asset_returns = prices.pct_change().fillna(0.0)
    equal_weights = np.ones(len(tickers)) / len(tickers)
    benchmark_returns = (daily_asset_returns * equal_weights).sum(axis=1)
    
    # vi) Calculate Institutional Metrics
    logging.info("\n=== Black-Litterman Strategy Metrics ===")
    bl_metrics = backtester.calculate_metrics(bl_strategy_returns)
    for key, value in bl_metrics.items():
        print(f"{key}: {value}")
        
    logging.info("\n=== Equal-Weight Benchmark Metrics ===")
    bench_metrics = backtester.calculate_metrics(benchmark_returns)
    for key, value in bench_metrics.items():
        print(f"{key}: {value}")

    # vii) Plot Cumulative Equity Curves
    logging.info("Generating performance plot...")
    
    bl_cumulative = (1 + bl_strategy_returns).cumprod()
    bench_cumulative = (1 + benchmark_returns).cumprod()
    
    plt.figure(figsize=(12, 6))
    plt.plot(bl_cumulative.index, bl_cumulative, label='Black-Litterman Optimized', color='blue', linewidth=2)
    plt.plot(bench_cumulative.index, bench_cumulative, label='Equal-Weight Benchmark', color='gray', linestyle='--')
    
    plt.title('Walk-Forward Backtest: Black-Litterman Fundamental Alpha vs Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 4. Run
if __name__ == "__main__":
    main()