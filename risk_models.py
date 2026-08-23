# Script 3: Calculate covariance with Ledoit-Wolf shrinkage.
# Calculate implied equilibrium returns to read the market's current baseline expectations

# 1. Imports
import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
import logging

# 2. Format Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 3. Define Risk Model Class for use in Main Script
class RiskModel:
    """
    Calculates robust covariance matrices and extracts market-implied 
    equilibrium returns for the Black-Litterman model.
    """
    def __init__(self, risk_aversion: float = 2.5):
        # Delta: The market risk aversion coefficient. 2.5 is standard in BL literature.
        self.risk_aversion = risk_aversion # represents how much return an average investor demands per unit of variance

    # 4. Define Covariance Calculator Function Including Ledoit-Wolf Shrinkage
    def calculate_covariance(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates an annualized robust covariance matrix using Ledoit-Wolf shrinkage.
        """
        logging.info("Calculating Ledoit-Wolf shrunken covariance matrix...")
        
        # Calculate daily log returns for numerical stability
        returns = np.log(prices / prices.shift(1)).dropna() # time additive and mathematically symmetric
        
        if returns.empty:
            raise ValueError("Returns dataframe is empty. Cannot calculate covariance.")
            
        # Fit Ledoit-Wolf estimator
        lw = LedoitWolf() # a standard sample covariance matrix is full of noise and spuriously high correlations. The convex optimiser to be used in script 5 will maximise these anomalies to create insane portfolio weights
        shrunk_cov = lw.fit(returns).covariance_ # mathematically shrink the sample matrix toward a highly structured stable matrix. Prevents our optimiser from breaking
        
        # Annualise the covariance matrix (assuming 252 trading days in a year)
        annualized_cov = shrunk_cov * 252 # make compatible with Q by multiplying daily covariance matrix by 252 trading days
        
        # Reconstruct DataFrame with original index and columns for ease of use
        cov_df = pd.DataFrame(annualized_cov, index=returns.columns, columns=returns.columns)
        
        return cov_df

    # 5. Define an implied returns calculator function
    def calculate_implied_returns(self, cov_matrix: pd.DataFrame, market_weights: pd.Series) -> pd.Series:
        """
        Reverse-optimizes the market portfolio to find implied equilibrium returns (Π).
        Formula: Π = δ * Σ * w_mkt
        """
        logging.info("Calculating implied equilibrium returns...")
        
        # Ensure indices align perfectly to prevent matrix multiplication errors
        market_weights = market_weights.reindex(cov_matrix.index).fillna(0.0) # defensive programming in case a stock was dropped due to NaN
        
        # Matrix multiplication: Π = δ * Σ * w_mkt
        implied_returns = self.risk_aversion * cov_matrix.dot(market_weights) # run CAPM in reverse. Assume current market cap weights of the S&P 500 are the optimal portfolio, then multiply those weights by the covariance matrix and risk aversion coeff
        
        return implied_returns

# 6. Testing
if __name__ == "__main__":
    # Test execution block
    np.random.seed(42)
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'BRK-B', 'JNJ', 'JPM', 'V', 'PG']
    
    # 1. Simulate 1 year of daily prices using Geometric Brownian Motion logic
    dates = pd.date_range(start="2025-01-01", periods=252, freq="B")
    simulated_returns = np.random.multivariate_normal(mean=np.zeros(10), cov=np.eye(10)*0.0001, size=252)
    simulated_prices = pd.DataFrame(simulated_returns, index=dates, columns=test_tickers)
    simulated_prices = 100 * np.exp(simulated_prices.cumsum())
    
    # 2. Simulate baseline market weights (e.g., market-cap weighting summing to 1)
    sim_weights = np.random.uniform(0.05, 0.15, 10)
    market_weights = pd.Series(sim_weights / sim_weights.sum(), index=test_tickers)
    
    risk_engine = RiskModel(risk_aversion=2.5)
    
    # Calculate Covariance
    cov_matrix = risk_engine.calculate_covariance(simulated_prices)
    print("\nAnnualized Ledoit-Wolf Covariance Matrix (Subset):")
    print(cov_matrix.iloc[:3, :3])
    
    # Calculate Implied Returns
    implied_returns = risk_engine.calculate_implied_returns(cov_matrix, market_weights)
    print("\nImplied Equilibrium Returns (Π):")
    print(implied_returns)