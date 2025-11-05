import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# --- Component 1, 2 & 3: Define Your Portfolio ---
# This is where you define your investor profile and security choices.
# This example uses a "Balanced" profile.
# NOTE: YOU MUST CHANGE THESE VALUES BASED ON YOUR OWN RESEARCH.

# 1. Define Initial Capital
INITIAL_CAPITAL = 1_000_000.00

# 2. Define Portfolio (Asset Allocation & Security Selection)
# Use tickers from Yahoo Finance (e.g., ".NS" for NSE stocks)
# The weights MUST sum to 1.0 (or 100%)
portfolio_weights = {
    'RELIANCE.NS': 0.15,   # Large-cap Equity
    'HDFCBANK.NS': 0.15,   # Large-cap Equity
    'TATAMOTORS.NS': 0.10, # Mid-cap Equity
    'INFY.NS': 0.10,       # IT Equity
    'GOLDBEES.NS': 0.20,   # Gold ETF (Hedge)
    'LIQUIDBEES.NS': 0.30  # Debt/Liquid ETF (Stability)
}

# 3. Define Benchmark
BENCHMARK_TICKER = '^NSEI' # NIFTY 50

# 4. Define Time Period for Analysis
# For a 4-week project, you'd set this to your project's start/end dates.
# For calculating long-term Beta/Sharpe, a 1-year period is better.
START_DATE = '2024-11-01'
# Use today's date for the end date
END_DATE = datetime.today().strftime('%Y-%m-%d')

# 5. Define Assumed Risk-Free Rate (e.g., T-Bill or FD rate)
RISK_FREE_RATE = 0.06 # 6% annual rate

# ---------------------------------------------------
# --- Component 4: Performance Tracking Engine ---
# ---------------------------------------------------

def get_stock_data(tickers, start, end):
    """Downloads 'Adj Close' price data from Yahoo Finance."""
    try:
        data = yf.download(tickers, start=start, end=end)['Adj Close']
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def calculate_portfolio_performance(data, weights, benchmark_ticker):
    """Calculates daily returns, cumulative returns, and key metrics."""
    
    # 1. Calculate Daily Returns for all assets
    # .pct_change() calculates the percentage change from the previous row
    daily_returns = data.pct_change().dropna()
    
    # 2. Separate benchmark returns from asset returns
    benchmark_returns = daily_returns[benchmark_ticker]
    asset_returns = daily_returns.drop(columns=[benchmark_ticker])
    
    # 3. Calculate Weighted Portfolio Daily Returns
    # This is the core of portfolio calculation: (Return_Asset1 * Weight_1) + (Return_Asset2 * Weight_2) + ...
    # We use the list of weights in the same order as the asset columns
    portfolio_asset_weights = [weights[col] for col in asset_returns.columns]
    portfolio_returns = (asset_returns * portfolio_asset_weights).sum(axis=1)
    
    # 4. Calculate Cumulative Returns
    # (1 + daily_return).cumprod() shows how the portfolio grows over time
    portfolio_cumulative_returns = (1 + portfolio_returns).cumprod()
    benchmark_cumulative_returns = (1 + benchmark_returns).cumprod()
    
    return portfolio_returns, benchmark_returns, portfolio_cumulative_returns, benchmark_cumulative_returns

def calculate_performance_metrics(portfolio_returns, benchmark_returns, risk_free_rate):
    """Calculates annualized metrics: Return, Volatility, Sharpe Ratio, and Beta."""
    
    # Assuming 252 trading days in a year
    trading_days = 252
    
    # --- Annualized Return ---
    mean_daily_return = portfolio_returns.mean()
    annual_return = (1 + mean_daily_return)**trading_days - 1
    
    # --- Annualized Volatility (Standard Deviation) ---
    annual_volatility = portfolio_returns.std() * np.sqrt(trading_days)
    
    # --- Sharpe Ratio ---
    daily_rf_rate = (1 + risk_free_rate)**(1/trading_days) - 1
    excess_returns = portfolio_returns - daily_rf_rate
    
    # Handle potential division by zero if volatility is zero
    if excess_returns.std() == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(trading_days)
        
    # --- Beta ---
    # Beta = Covariance(Portfolio, Market) / Variance(Market)
    # Create a DataFrame to ensure alignment before calculating covariance
    returns_df = pd.DataFrame({
        'portfolio': portfolio_returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    covariance_matrix = np.cov(returns_df['portfolio'], returns_df['benchmark'])
    covariance = covariance_matrix[0, 1]
    market_variance = returns_df['benchmark'].var()
    
    beta = covariance / market_variance
    
    return annual_return, annual_volatility, sharpe_ratio, beta

def plot_performance(portfolio_cumulative, benchmark_cumulative, initial_capital, benchmark_ticker):
    """Plots the portfolio value vs. benchmark value over time."""
    
    portfolio_value = portfolio_cumulative * initial_capital
    benchmark_value = benchmark_cumulative * initial_capital
    
    plt.figure(figsize=(12, 6))
    
    # Plot portfolio value
    plt.plot(portfolio_value.index, portfolio_value, label='My Portfolio', color='blue', linewidth=2)
    
    # Plot benchmark value
    plt.plot(benchmark_value.index, benchmark_value, label=f'Benchmark ({benchmark_ticker})', color='grey', linestyle='--')
    
    plt.title(f'Portfolio Performance vs. {benchmark_ticker}')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (₹)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Format Y-axis to show "₹"
    from matplotlib.ticker import FuncFormatter
    formatter = FuncFormatter(lambda x, p: f'₹{x:,.0f}')
    plt.gca().yaxis.set_major_formatter(formatter)
    
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting Portfolio Analysis for {list(portfolio_weights.keys())}...")
    
    # 1. Get Tickers
    all_tickers = list(portfolio_weights.keys()) + [BENCHMARK_TICKER]
    
    # 2. Fetch Data
    price_data = get_stock_data(all_tickers, START_DATE, END_DATE)
    
    if price_data is not None and not price_data.empty:
        # 3. Calculate Performance
        port_returns, bench_returns, port_cum_returns, bench_cum_returns = \
            calculate_portfolio_performance(price_data, portfolio_weights, BENCHMARK_TICKER)
        
        # 4. Calculate Metrics
        ann_return, ann_vol, sharpe, beta = \
            calculate_performance_metrics(port_returns, bench_returns, RISK_FREE_RATE)
        
        # 5. Get Final Portfolio Value
        final_portfolio_value = port_cum_returns.iloc[-1] * INITIAL_CAPITAL
        total_return_pct = (final_portfolio_value / INITIAL_CAPITAL) - 1
        
        # 6. Print Summary Report (Components 4 & 6)
        print("\n--- PORTFOLIO PERFORMANCE REPORT ---")
        print(f"Period: {START_DATE} to {END_DATE}\n")
        print(f"Initial Capital:  ₹{INITIAL_CAPITAL:,.2f}")
        print(f"Final Value:      ₹{final_portfolio_value:,.2f}")
        print(f"Total Return:     {total_return_pct:.2%}\n")
        
        print("--- Key Metrics (Annualized) ---")
        print(f"Annualized Return:      {ann_return:.2%}")
        print(f"Annualized Volatility:  {ann_vol:.2%}")
        print(f"Sharpe Ratio:           {sharpe:.2f}")
        print(f"Beta vs. {BENCHMARK_TICKER}:      {beta:.2f}")
        
        # 7. Plot Graph
        plot_performance(port_cum_rules, bench_cum_returns, INITIAL_CAPITAL, BENCHMARK_TICKER)
        
    else:
        print("Could not retrieve data to perform analysis.")
