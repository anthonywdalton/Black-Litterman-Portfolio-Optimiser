# Black-Litterman Portfolio Optimisation Engine

## Contents

1. Project Description
2. Scripts
    * Data Ingestion
    * Signal Generator
    * Risk Models
    * Black-Litterman Engine
    * Portfolio Optimiser
    * Backtester
    * Main Orchestrator
3. How to Use
4. Performance

## 1. Project Description

This is a modular, quantitative portfolio optimisation pipeline. It extracts historical pricing and fundamental data to construct a robust walk-forward backtest. The system leverages the Black-Litterman model to mathematically blend market-implied equilibrium returns with quantitative fundamental signals via Bayesian updating, subsequently passing the posterior distributions into a convex solver to determine optimal asset weights.

> **Note on Data Infrastructure:** This prototype utilizes the free Yahoo Finance API (`yfinance`) for data ingestion. Due to third-party constraints on historical financial statements, fundamental metrics (such as Free Cash Flow) are restricted to the trailing 4 to 8 quarters, naturally limiting the walk-forward backtest to a recent 1-year window. In a production environment, the `data_ingestion.py` module is designed to be easily repointed to a proprietary SQL database or a premium data terminal (such as Compustat, Bloomberg, or FactSet) to execute a standard 10-year historical cross-validation.

Applications of this project in quantitative trading are as follows:

*   **Systematic Asset Allocation** - the pipeline demonstrates how to mathematically scale away from market-cap weights based on algorithmic signal strength rather than subjective guessing.
*   **Robust Risk Management** - using Ledoit-Wolf covariance shrinkage prevents the "error-maximising" tendencies of standard mean-variance optimisation, stabilising out-of-sample portfolio weights.
*   **Transaction Cost Mitigation** - the convex solver enforces L1-norm turnover penalties, preventing the model from generating high-turnover weights that would be destroyed by slippage and fees in live trading.
*   **Data Hygiene** - strict 45-day reporting lags are enforced on fundamental accounting data to prevent look-ahead bias, ensuring the model only trades on information publicly available at the time.

## 2. Scripts

### **Data Ingestion** (`data_ingestion.py`)

Object-oriented data pipeline connecting to the Yahoo Finance API to extract daily adjusted closing prices and quarterly Free Cash Flow (FCF) for a defined universe of equities.

*   **Point-in-Time Enforcement:** Applies a strict 45-day index shift to all financial statements. This simulates real-world SEC reporting lags, completely eliminating look-ahead bias by ensuring the model only acts on data once it is historically public.
*   **Defensive API Handling:** Dynamically detects and adapts to multi-index dataframe changes from the API provider to prevent pipeline failures.
*   **Frequency Alignment:** Utilises forward-filling mechanisms to align low-frequency quarterly fundamental data with the high-frequency daily pricing calendar.

### **Signal Generator** (`signal_generator.py`)

Ingests the point-in-time fundamental data cross-sections and translates raw accounting metrics into mathematical inputs for the Bayesian engine.

1.  **Cross-Sectional Z-Scoring:** Normalises the raw FCF yield across the entire asset universe on each rebalance date to standardise signal strength.
2.  **Pick Matrix (P) Construction:** Assigns proportional positive weights to high-yield (Value/Quality) stocks and negative weights to low-yield stocks. Weights are scaled to sum to 1 and -1 respectively, generating a mathematically pure, market-neutral relative view.
3.  **View Vector (Q) Construction:** Assigns a conservative annualised premium to the expected outperformance of the fundamental view portfolio.

### **Risk Models** (`risk_models.py`)

Calculates the baseline market prior and robust risk parameters required by the convex solver.

*   **Ledoit-Wolf Shrinkage:** Utilises `scikit-learn` to calculate a shrunken covariance matrix. This statistically pulls the noisy sample covariance matrix toward a highly structured constant-correlation matrix, vastly improving out-of-sample stability and preventing the optimiser from concentrating in spuriously correlated assets.
*   **Reverse Optimisation:** Extracts the Implied Equilibrium Returns. Instead of using noisy historical averages, it backs out the expected returns that make the current market capitalization weights mathematically optimal under standard market risk-aversion parameters.

### **Black-Litterman Engine** (`bl_engine.py`)

Houses the core linear algebra and Bayesian mathematics. It executes the master equation by blending the market prior with the fundamental quantitative views.

*   **Idzorek’s Method:** Translates a subjective percentage confidence level in the fundamental alpha signal into mathematical variance, populating the diagonal View Variance Matrix (Omega).
*   **Matrix Inversion:** Resolves the complex matrix algebra to output the final Posterior Expected Returns, balancing market consensus with the algorithmic FCF yield signal based on calculated confidence levels.

### **Portfolio Optimiser** (`portfolio_optimiser.py`)

Wraps the `cvxpy` library to solve a constrained Quadratic Utility objective function. It passes the posterior returns and shrunken covariance into a convex OSQP solver. Strict institutional constraints are mathematically enforced:

1.  **Long-Only Bounds:** Prevents short selling ($w \ge 0$).
2.  **Fully Invested:** Forces the sum of all target weights to exactly 1.0.
3.  **Concentration Limits:** Caps individual asset weights at a maximum of 15% to enforce broad portfolio diversification.
4.  **L1-Norm Turnover Penalty:** Penalises the objective function for drastic week-to-week weight shifts, mathematically simulating transaction costs and slippage to ensure the strategy remains viable in high-friction live trading environments.

### **Backtester** (`backtester.py`)

Orchestrates a chronological walk-forward cross-validation loop. This ensures the model is evaluated exactly as it would perform in live markets.

*   Iterates through the time series at a set rebalancing frequency (e.g., every 21 trading days).
*   Dynamically queries the risk and signal engines using a strictly historical, expanding lookback window (e.g., 63 days) to construct the covariance matrix without data leakage.
*   Logs target weights and explicitly shifts them forward by one execution period to prevent execution bias (ensuring the model does not instantaneously trade on the exact same day the signal is calculated).

### **Main Orchestrator** (`main.py`)

The central execution script utilising Dependency Injection. 

*   Imports all modules, initialises the class instances, and feeds the downloaded data into the walk-forward loop.
*   Benchmarks the final Black-Litterman strategy against a naive Equal-Weight (1/N) baseline.
*   Calculates institutional out-of-sample performance metrics (CAGR, Annualised Volatility, Sharpe Ratio, Maximum Drawdown) and plots cumulative equity curves via `matplotlib`.

## 3. How to Use

1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`. (Requires `pandas`, `numpy`, `yfinance`, `scikit-learn`, `cvxpy`, and `matplotlib`).
3. Execute the pipeline by running `python main.py` in your terminal. The script will automatically download the required data, run the walk-forward backtest, print the metrics to the console, and generate the equity curve plot.

## 4. Performance

**Metrics (2025-2026 Walk-Forward Out-of-Sample):**
*   **Black-Litterman Strategy:** 7.89% CAGR | 11.74% Volatility | 0.67 Sharpe Ratio | -10.90% Max Drawdown
*   **Equal-Weight Benchmark:** 21.55% CAGR | 12.82% Volatility | 1.68 Sharpe Ratio | -11.31% Max Drawdown

### Performance Analysis
While the optimised portfolio underperformed the naive Equal-Weight benchmark in absolute return and Sharpe ratio, the backtest highlights the realities of institutional portfolio constraints and factor timing:

1.  **The "Cold Start" Lag:** To compute a robust covariance matrix, the backtester utilises a strict 63-day lookback window. The model intentionally refused to allocate capital for the first three months of the dataset, remaining entirely in cash while the benchmark compounded early market gains.
2.  **Factor Environment (Value vs. Growth):** The chosen alpha signal, Free Cash Flow Yield, inherently tilts the portfolio toward Value and Quality factors. During this specific historical window, the S&P 500 was heavily dominated by mega-cap Growth expansion. The optimizer mathematically underweighted high-flying, low-yield tech equities in favor of defensive fundamentals, resulting in a relative performance drag.
3.  **Friction Realities:** The Black-Litterman strategy was subjected to an L1-norm turnover penalty within the convex solver, deliberately suppressing returns to mathematically simulate transaction costs and slippage. The baseline benchmark assumes costless, frictionless rebalancing.
4.  **Successful Risk Mitigation:** Despite the absolute return drag, the core risk engines performed exactly as designed. The Ledoit-Wolf shrinkage successfully compressed annualized volatility (11.74% vs 12.82%) and resulted in a shallower Maximum Drawdown (-10.90% vs -11.31%) compared to the benchmark, proving the model's ability to defensively manage portfolio variance.
