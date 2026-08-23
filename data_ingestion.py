import yfinance as yf
import pandas as pd
import numpy as np
import logging
import os
import tempfile

# 1. Suppress Windows TzCache warning by routing it to a temporary directory
yf.set_tz_cache_location(os.path.join(tempfile.gettempdir(), "yfinance_tz_cache"))

# Configure basic logging for professional terminal output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataIngestion:
    """
    Handles fetching and formatting pricing and fundamental data for the 
    Black-Litterman optimization pipeline, enforcing strict point-in-time reporting.
    """
    
    def __init__(self, tickers: list, start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.reporting_lag = 45 # Days after quarter-end before data is 'public'

    def fetch_pricing_data(self) -> pd.DataFrame:
        """
        Downloads adjusted closing prices with defensive fallback mechanisms.
        """
        logging.info(f"Downloading daily pricing data for {len(self.tickers)} assets...")
        
        # Download data
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date, progress=False)
        
        if data.empty:
            raise ValueError("Pricing data download failed. Check ticker symbols and dates.")
            
        # Defensive Programming: Handle yfinance API structure changes dynamically
        if isinstance(data.columns, pd.MultiIndex):
            # If multiple tickers, columns are a MultiIndex (e.g., 'Adj Close', 'AAPL')
            top_level_columns = data.columns.levels[0]
            if 'Adj Close' in top_level_columns:
                prices = data['Adj Close']
            elif 'Close' in top_level_columns:
                logging.info("'Adj Close' not found. Using pre-adjusted 'Close' prices.")
                prices = data['Close']
            else:
                raise KeyError(f"Expected 'Adj Close' or 'Close'. Found: {top_level_columns.tolist()}")
        else:
            # Fallback for single ticker or flat structure
            if 'Adj Close' in data.columns:
                prices = data['Adj Close']
            elif 'Close' in data.columns:
                prices = data['Close']
            else:
                raise KeyError(f"Expected 'Adj Close' or 'Close'. Found: {data.columns.tolist()}")
            
        # Forward-fill missing prices (e.g., due to trading halts), then drop early NAs
        prices = prices.ffill().dropna(how='all')
        
        logging.info("Pricing data successfully ingested.")
        return prices

    def fetch_fundamental_fcf_yield(self, adj_close: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts quarterly Free Cash Flow, applies a 45-day reporting lag to 
        prevent look-ahead bias, and merges it against the daily price index.
        """
        logging.info("Downloading fundamental cash flow data...")
        
        fcf_yield_df = pd.DataFrame(index=adj_close.index, columns=self.tickers)
        
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                
                # Fetch quarterly cash flow and transpose so dates are the index
                qcf = stock.quarterly_cashflow.T
                
                if qcf.empty or 'Free Cash Flow' not in qcf.columns:
                    logging.warning(f"Free Cash Flow data missing for {ticker}. Skipping.")
                    continue
                    
                fcf = qcf[['Free Cash Flow']].copy()
                fcf.index = pd.to_datetime(fcf.index)
                
                # Sort chronologically
                fcf = fcf.sort_index()
                
                # APPLY REPORTING LAG: Shift the index by 45 days to simulate publication date
                fcf.index = fcf.index + pd.Timedelta(days=self.reporting_lag)
                
                # Reindex to the daily pricing calendar and forward-fill the lagged fundamental data
                fcf_daily = fcf.reindex(adj_close.index).ffill()
                
                # Calculate Market Proxy for EV using historical price * current shares outstanding
                shares_out = stock.info.get('sharesOutstanding', np.nan)
                if pd.isna(shares_out) or shares_out == 0:
                    continue
                    
                market_cap_proxy = adj_close[ticker] * shares_out
                
                # Calculate FCF Yield
                fcf_yield_df[ticker] = fcf_daily['Free Cash Flow'] / market_cap_proxy
                
            except Exception as e:
                logging.error(f"Error processing fundamentals for {ticker}: {e}")

        # Final clean-up: forward fill any intermediate NAs and drop columns that are entirely NA
        fcf_yield_df = fcf_yield_df.ffill().dropna(axis=1, how='all')
        
        logging.info("Point-in-time FCF Yield successfully generated.")
        return fcf_yield_df

if __name__ == "__main__":
    # Test execution block
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'BRK-B', 'JNJ', 'JPM', 'V', 'PG']
    
    ingestion_engine = DataIngestion(
        tickers=test_tickers,
        start_date="2025-08-01", # Changed to a recent date
        end_date="2026-08-20"    # Changed to current timeframe
    )
    
    # 1. Fetch Prices
    prices = ingestion_engine.fetch_pricing_data()
    print("\nSample Pricing Data:")
    print(prices.tail())
    
    # 2. Fetch Fundamentals and Calculate Lagged FCF Yield
    fcf_yields = ingestion_engine.fetch_fundamental_fcf_yield(prices)
    print("\nSample FCF Yield Data (Point-in-Time):")
    print(fcf_yields.tail())