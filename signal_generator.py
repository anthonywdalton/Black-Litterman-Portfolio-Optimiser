# Script 2: 

# 1. Imports
import pandas as pd
import numpy as np
import logging

# 2. Format Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 3. Define Signal Generator Class for use in Main Script
class SignalGenerator:
    """
    Translates raw fundamental data into quantitative Black-Litterman views.
    """
    def __init__(self, expected_outperformance: float = 0.03):
        # The annualized expected return premium of the fundamental view - default 3%
        self.expected_outperformance = expected_outperformance

    # 4. Define View Generator Function that takes ingested data and returns P and Q
    def generate_views(self, current_fcf_yields: pd.Series):
        """
        Takes a point-in-time cross-section of FCF yields and returns 
        the P matrix (Pick Matrix) and Q vector (View Expected Return).
        """
        # Drop assets with missing fundamentals for this specific period
        valid_yields = current_fcf_yields.dropna() # remove NaNs
        
        if len(valid_yields) < 2: # must have at least 2 valid stocks from data feed so that long weights sum to 1 and shorts to -1.
            logging.warning("Not enough valid fundamental data to generate views.")
            return None, None
            
        # i) Cross-Sectional Z-Score Calculation (use z scores as we are interested in relative winners)
        z_scores = (valid_yields - valid_yields.mean()) / valid_yields.std() # standardise signal, neutralising broad market shifts
        
        # ii) Construct the Pick Matrix (P)
        # Initialize a zero-weight series for the 50 stock universe
        P_series = pd.Series(0.0, index=current_fcf_yields.index) # if a stock has no signal, its weight in the P matrix is 0
        
        pos_z = z_scores[z_scores > 0] # candidates to long
        neg_z = z_scores[z_scores < 0] # candidates to short
        
        # Proportional weighting based on signal strength: long high-yield, short low-yield
        if not pos_z.empty and not neg_z.empty:
            P_series[pos_z.index] = pos_z / pos_z.sum() # a stock with massive FCF yield gets a larger positive weight in the view
            P_series[neg_z.index] = neg_z / abs(neg_z.sum()) # repeat for shorts. long side of view adds to 1 and short side to -1 to be market neutral
            
        # Convert to 2D numpy array (1 x N)
        P_matrix = P_series.values.reshape(1, -1) # B-L master equation relies on linear algebra so must be matrix form
        
        # iii) Construct the View Vector (Q)
        # 1D numpy array representing the expected outperformance
        Q_vector = np.array([self.expected_outperformance]) # again this value must be in matrix form for the linear algebra occuring in bl_engine.py
        
        return P_matrix, Q_vector

# 5. Testing
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