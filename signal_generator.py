import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SignalGenerator:
    """
    Translates raw fundamental data into quantitative Black-Litterman views.
    """
    def __init__(self, expected_outperformance: float = 0.03):
        # The annualized expected return premium of the fundamental view
        self.expected_outperformance = expected_outperformance

    def generate_views(self, current_fcf_yields: pd.Series):
        """
        Takes a point-in-time cross-section of FCF yields and returns 
        the P matrix (Pick Matrix) and Q vector (View Expected Return).
        """
        # Drop assets with missing fundamentals for this specific period
        valid_yields = current_fcf_yields.dropna()
        
        if len(valid_yields) < 2:
            logging.warning("Not enough valid fundamental data to generate views.")
            return None, None
            
        # 1. Cross-Sectional Z-Score Calculation
        z_scores = (valid_yields - valid_yields.mean()) / valid_yields.std()
        
        # 2. Construct the Pick Matrix (P)
        # Initialize a zero-weight series matching the original universe index
        P_series = pd.Series(0.0, index=current_fcf_yields.index)
        
        pos_z = z_scores[z_scores > 0]
        neg_z = z_scores[z_scores < 0]
        
        # Proportional weighting: long high-yield, short low-yield
        if not pos_z.empty and not neg_z.empty:
            P_series[pos_z.index] = pos_z / pos_z.sum()
            P_series[neg_z.index] = neg_z / abs(neg_z.sum())
            
        # Convert to 2D numpy array (1 x N)
        P_matrix = P_series.values.reshape(1, -1)
        
        # 3. Construct the View Vector (Q)
        # 1D numpy array representing the expected outperformance
        Q_vector = np.array([self.expected_outperformance])
        
        return P_matrix, Q_vector

if __name__ == "__main__":
    # Test execution block simulating a single day's cross-section
    np.random.seed(42)
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'BRK-B', 'JNJ', 'JPM', 'V', 'PG']
    
    # Simulate some point-in-time FCF yields (using random normal data for the test)
    simulated_yields = pd.Series(np.random.normal(0.05, 0.02, len(test_tickers)), index=test_tickers)
    
    print("Simulated FCF Yields:")
    print(simulated_yields)
    
    generator = SignalGenerator(expected_outperformance=0.03)
    P, Q = generator.generate_views(simulated_yields)
    
    print("\nPick Matrix (P):")
    # Zipping tickers with their P-matrix weights for readability
    print(dict(zip(test_tickers, np.round(P[0], 4))))
    
    print(f"\nView Vector (Q): {Q}")