import pandas as pd
import numpy as np
import cvxpy as cp
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PortfolioOptimizer:
    """
    Constructs and solves convex optimization problems to find optimal 
    portfolio weights given expected returns and covariance.
    """
    def __init__(self, risk_aversion: float = 2.5, turnover_penalty: float = 0.001):
        self.risk_aversion = risk_aversion
        self.turnover_penalty = turnover_penalty # Gamma (γ)
        self.max_weight = 0.15 # 15% position limit

    def optimize_weights(self, 
                         expected_returns: pd.Series, 
                         cov_matrix: pd.DataFrame, 
                         previous_weights: pd.Series = None) -> pd.Series:
        """
        Maximizes quadratic utility subject to long-only, fully-invested, 
        and L1-norm turnover constraints.
        """
        logging.info("Initializing cvxpy convex solver...")
        
        n_assets = len(expected_returns)
        mu = expected_returns.values
        Sigma = cov_matrix.values
        
        # Initialize cvxpy variable for weights
        w = cp.Variable(n_assets)
        
        # 1. Define the Objective Function: Maximize Risk-Adjusted Return
        portfolio_return = w.T @ mu
        portfolio_risk = cp.quad_form(w, Sigma)
        
        objective_expr = portfolio_return - (self.risk_aversion / 2) * portfolio_risk
        
        # 2. Add L1-Norm Turnover Penalty (if previous weights exist)
        if previous_weights is not None:
            w_prev = previous_weights.reindex(expected_returns.index).fillna(0.0).values
            turnover = cp.norm(w - w_prev, 1)
            objective_expr -= self.turnover_penalty * turnover
            
        objective = cp.Maximize(objective_expr)
        
        # 3. Define Constraints
        constraints = [
            cp.sum(w) == 1,           # Fully invested
            w >= 0,                   # Long-only
            w <= self.max_weight      # Maximum position limit
        ]
        
        # 4. Solve the Problem
        prob = cp.Problem(objective, constraints)
        
        try:
            # OSQP is a highly robust solver for Quadratic Programs
            prob.solve(solver=cp.OSQP)
        except Exception as e:
            logging.error(f"Solver failed: {e}")
            return None
            
        if prob.status != cp.OPTIMAL:
            logging.warning(f"Solver did not find optimal solution. Status: {prob.status}")
            
        # Clean up near-zero weights due to floating point math
        optimal_weights = np.array(w.value)
        optimal_weights[optimal_weights < 1e-4] = 0.0
        
        # Normalize just in case of rounding errors summing to 0.9999
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        # Reconstruct pandas Series
        weights_series = pd.Series(optimal_weights, index=expected_returns.index)
        
        logging.info("Optimal weights successfully calculated.")
        return weights_series

if __name__ == "__main__":
    # Test execution block
    np.random.seed(42)
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'BRK-B', 'JNJ', 'JPM', 'V', 'PG']
    
    # Mock posterior returns and covariance
    mock_er = pd.Series(np.random.normal(0.08, 0.02, 10), index=test_tickers)
    
    # Create a positive semi-definite mock covariance matrix
    A = np.random.rand(10, 10)
    mock_cov = pd.DataFrame(np.dot(A, A.transpose()) * 0.001, index=test_tickers, columns=test_tickers)
    
    # Mock previous weights (e.g., from last month's rebalance)
    mock_prev_w = pd.Series(0.10, index=test_tickers) 
    
    optimizer = PortfolioOptimizer()
    optimal_w = optimizer.optimize_weights(mock_er, mock_cov, previous_weights=mock_prev_w)
    
    print("\nOptimal Target Portfolio Weights:")
    print(optimal_w.round(4))
    print(f"\nTotal Sum: {optimal_w.sum():.2f}")