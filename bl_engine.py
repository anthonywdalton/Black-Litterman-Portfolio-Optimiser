import pandas as pd
import numpy as np
from numpy.linalg import inv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BlackLittermanEngine:
    """
    Houses the core Bayesian mathematics for the Black-Litterman model, 
    blending market priors with quantitative views.
    """
    def __init__(self, tau: float = 0.05, confidence: float = 0.65):
        # Tau: Scalar representing uncertainty in the CAPM prior (usually 0.01 - 0.05)
        self.tau = tau
        # Confidence: Percentage confidence in our fundamental view (Idzorek's method)
        self.confidence = confidence

    def compute_posterior_returns(self, 
                                  implied_returns: pd.Series, 
                                  cov_matrix: pd.DataFrame, 
                                  P_matrix: np.ndarray, 
                                  Q_vector: np.ndarray) -> pd.Series:
        """
        Calculates the posterior expected returns vector E[R] using the BL Master Equation.
        """
        logging.info("Executing Black-Litterman Bayesian update...")
        
        # Convert pandas structures to numpy arrays for linear algebra
        Pi = implied_returns.values.reshape(-1, 1)
        Sigma = cov_matrix.values
        P = P_matrix
        Q = Q_vector.reshape(-1, 1)
        
        # 1. Calculate baseline Omega (Variance of the views)
        # Formula: P * (tau * Sigma) * P_transpose
        omega_baseline = np.dot(np.dot(P, self.tau * Sigma), P.T)
        
        # 2. Apply Idzorek's Method to scale Omega based on percentage confidence
        # If confidence is 100%, Omega approaches 0 (absolute certainty).
        # If confidence is low, Omega approaches infinity (view is ignored).
        if self.confidence >= 1.0 or self.confidence <= 0.0:
            raise ValueError("Confidence must be strictly between 0 and 1 exclusive.")
            
        alpha = (1 - self.confidence) / self.confidence
        Omega = omega_baseline * alpha
        
        # Ensure Omega is a diagonal matrix (though we only have 1 view here, it's best practice)
        Omega = np.diag(np.diag(Omega))
        
        # 3. Master Equation Matrix Inversions
        tau_sigma_inv = inv(self.tau * Sigma)
        omega_inv = inv(Omega)
        
        # Left term: [(tau * Sigma)^-1 + P^T * Omega^-1 * P]^-1
        left_term = inv(tau_sigma_inv + np.dot(np.dot(P.T, omega_inv), P))
        
        # Right term: [(tau * Sigma)^-1 * Pi + P^T * Omega^-1 * Q]
        right_term = np.dot(tau_sigma_inv, Pi) + np.dot(np.dot(P.T, omega_inv), Q)
        
        # E[R] = Left_term * Right_term
        posterior_expected_returns = np.dot(left_term, right_term)
        
        # Reconstruct into a pandas Series for the optimizer
        posterior_series = pd.Series(
            posterior_expected_returns.flatten(), 
            index=implied_returns.index
        )
        
        logging.info("Posterior expected returns successfully generated.")
        return posterior_series

if __name__ == "__main__":
    # Test execution block bridging the previous modules
    np.random.seed(42)
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'BRK-B', 'JNJ', 'JPM', 'V', 'PG']
    
    # 1. Mock inputs from RiskModel
    mock_pi = pd.Series(np.random.normal(0.06, 0.01, 10), index=test_tickers)
    mock_sigma = pd.DataFrame(np.eye(10)*0.02, index=test_tickers, columns=test_tickers)
    
    # 2. Mock inputs from SignalGenerator (1 view across 10 assets)
    mock_P = np.random.normal(0, 0.1, (1, 10))
    mock_P = mock_P / np.sum(np.abs(mock_P)) # Normalize loosely
    mock_Q = np.array([0.03])
    
    print("Prior Implied Returns (Π):")
    print(mock_pi.head(3))
    
    # 3. Run the BL Engine
    bl = BlackLittermanEngine(tau=0.05, confidence=0.65)
    posterior_returns = bl.compute_posterior_returns(mock_pi, mock_sigma, mock_P, mock_Q)
    
    print("\nPosterior Expected Returns (E[R]):")
    print(posterior_returns.head(3))