# Script 6: Backtester - chronological simulation engine

# 1. Imports
import pandas as pd
import numpy as np
import logging

# 2. Format Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 3. Define a walk-forward backtester class for use in main script
class WalkForwardBacktester:
    """
    Simulates a historical walk-forward backtest, rebalancing the portfolio 
    at set intervals without look-ahead bias, and calculates performance metrics.
    """
    def __init__(self, rebalance_freq: int = 21, lookback_window: int = 63):
        # 21 trading days ~ 1 month. 63 days ~ 3 months lookback for covariance
        self.rebalance_freq = rebalance_freq
        self.lookback_window = lookback_window

    # 4. Define a run backtest function
    def run_backtest(self, 
                     prices: pd.DataFrame, 
                     fcf_yields: pd.DataFrame, 
                     risk_engine, 
                     signal_engine, 
                     bl_engine, 
                     optimizer) -> pd.Series:
        """
        Executes the walk-forward loop across the dataset.
        """
        logging.info("Initializing Walk-Forward Backtester...")
        
        # Pre-calculate daily asset returns
        daily_asset_returns = prices.pct_change().fillna(0.0) # will multiply by weights at end to get PnL
        
        historical_weights = pd.DataFrame(index=prices.index, columns=prices.columns).fillna(0.0) # dataFrame to store our historical target weights
        current_weights = pd.Series(0.0, index=prices.columns) # tracks active positions to be passed into optimiser to calculate L1-norm turnover penalty
        
        # Walk-forward loop
        for i in range(self.lookback_window, len(prices), self.rebalance_freq): # starts at day 63 - results in cold start lag. Steo forward by 21 days each iteration
            current_date = prices.index[i]
            logging.info(f"Rebalancing for date: {current_date.date()}")
            
            # i) Isolate point-in-time data (strictly trailing to avoid look-ahead)
            price_window = prices.iloc[i - self.lookback_window : i] # prevent look-ahead
            current_fcf = fcf_yields.iloc[i]
            
            # ii) Risk Model
            cov_matrix = risk_engine.calculate_covariance(price_window) # pass point-in-time data 
            
            # (Simplification: using equal weight as baseline market proxy for the backtest)
            market_weights = pd.Series(1.0 / len(prices.columns), index=prices.columns)
            implied_returns = risk_engine.calculate_implied_returns(cov_matrix, market_weights)
            
            # iii) Signal Generation
            P, Q = signal_engine.generate_views(current_fcf)
            
            if P is None: # failsafe from signal generator
                logging.warning(f"Skipping rebalance on {current_date.date()} due to missing fundamental signals.")
                historical_weights.iloc[i] = current_weights # log current weights
                continue
                
            # 4. Black-Litterman Engine
            posterior_returns = bl_engine.compute_posterior_returns(implied_returns, cov_matrix, P, Q)
            
            # 5. Convex Optimization
            new_weights = optimizer.optimize_weights(posterior_returns, cov_matrix, previous_weights=current_weights)
            
            if new_weights is not None:
                current_weights = new_weights # overwrite with newly optimised target portfolio
                
            # Store the active weights for this period
            historical_weights.iloc[i] = current_weights

        # Forward-fill weights to represent holding the portfolio between rebalances
        # Shift by 1 period so we don't trade on the same day we calculate the signal. We have to wait until tomorrow's open
        historical_weights = historical_weights.replace(0.0, np.nan).ffill().fillna(0.0).shift(1).fillna(0.0)
        
        # Calculate daily portfolio returns: sum of (weights * asset returns)
        portfolio_daily_returns = (historical_weights * daily_asset_returns).sum(axis=1)
        
        return portfolio_daily_returns

    # 5. Define a function to calculate sharpe ratio and drawdowns
    def calculate_metrics(self, portfolio_returns: pd.Series) -> dict:
        """
        Calculates institutional performance metrics.
        """
        logging.info("Calculating performance metrics...")
        
        # Annualized Return
        cumulative_return = float((1 + portfolio_returns).prod()) - 1
        years = len(portfolio_returns) / 252
        cagr = (1 + cumulative_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Annualized Volatility
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Sharpe Ratio (Assuming 0% risk-free rate for prototype)
        sharpe_ratio = cagr / volatility if volatility > 0 else 0
        
        # Maximum Drawdown
        cumulative_index = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_index.cummax()
        drawdowns = (cumulative_index - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        metrics = {
            "CAGR": round(cagr, 4),
            "Volatility": round(volatility, 4),
            "Sharpe Ratio": round(sharpe_ratio, 4),
            "Max Drawdown": round(max_drawdown, 4)
        }
        
        return metrics