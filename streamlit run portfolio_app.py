import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date

# --- Core Portfolio Analysis Functions (from original script) ---
# These are the same functions as before, with one change:
# plot_performance now *returns* the figure object instead of showing it.

def get_stock_data(tickers, start, end):
    """Downloads 'Adj Close' price data from Yahoo Finance."""
    try:
        data = yf.download(tickers, start=start, end=end)['Adj Close']
        # Handle single-ticker download (returns a Series)
        if isinstance(data, pd.Series):
            data = data.to_frame(tickers[0])
        return data
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None

def calculate_portfolio_performance(data, weights, benchmark_ticker):
    """Calculates daily returns, cumulative returns, and key metrics."""
    
    # 1. Calculate Daily Returns for all assets
    daily_returns = data.pct_change().dropna()
    
    # 2. Separate benchmark returns from asset returns
    # Check if benchmark is in columns (it might not be if it's the only ticker)
    if benchmark_ticker in daily_returns.columns:
        benchmark_returns = daily_returns[benchmark_ticker]
        asset_returns = daily_returns.drop(columns=[benchmark_ticker])
    else:
        # Handle case where only the benchmark was downloaded
        benchmark_returns = daily_returns.iloc[:, 0]
        # Create an empty df for asset returns if no other assets
        asset_returns = pd.DataFrame(index=daily_returns.index)

    # 3. Calculate Weighted Portfolio Daily Returns (if there are assets)
    if not asset_returns.empty:
        # Ensure weights align with asset columns
        aligned_weights = [weights.get(col, 0) for col in asset_returns.columns]
        portfolio_returns = (asset_returns * aligned_weights).sum(axis=1)
    else:
        # If no assets, portfolio return is 0
        portfolio_returns = pd.Series(0.0, index=daily_returns.index)

    # 4. Calculate Cumulative Returns
    portfolio_cumulative_returns = (1 + portfolio_returns).cumprod()
    benchmark_cumulative_returns = (1 + benchmark_returns).cumprod()
    
    return portfolio_returns, benchmark_returns, portfolio_cumulative_returns, benchmark_cumulative_returns

def calculate_performance_metrics(portfolio_returns, benchmark_returns, risk_free_rate):
    """Calculates annualized metrics: Return, Volatility, Sharpe Ratio, and Beta."""
    
    # Assuming 252 trading days in a year
    trading_days = 252
    
    # --- Annualized Return ---
    # Handle empty portfolio returns
    if portfolio_returns.empty or portfolio_returns.sum() == 0:
        annual_return = 0.0
        annual_volatility = 0.0
        sharpe_ratio = 0.0
    else:
        mean_daily_return = portfolio_returns.mean()
        annual_return = (1 + mean_daily_return)**trading_days - 1
        annual_volatility = portfolio_returns.std() * np.sqrt(trading_days)
        
        # --- Sharpe Ratio ---
        daily_rf_rate = (1 + risk_free_rate)**(1/trading_days) - 1
        excess_returns = portfolio_returns - daily_rf_rate
        
        if excess_returns.std() == 0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(trading_days)
        
    # --- Beta ---
    returns_df = pd.DataFrame({
        'portfolio': portfolio_returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    # Handle cases with insufficient data for covariance
    if len(returns_df) < 2:
        beta = 0.0
    else:
        covariance_matrix = np.cov(returns_df['portfolio'], returns_df['benchmark'])
        covariance = covariance_matrix[0, 1]
        market_variance = returns_df['benchmark'].var()
        
        if market_variance == 0:
            beta = 0.0
        else:
            beta = covariance / market_variance
    
    return annual_return, annual_volatility, sharpe_ratio, beta

def plot_performance(portfolio_cumulative, benchmark_cumulative, initial_capital, benchmark_ticker):
    """Plots the portfolio value vs. benchmark value over time.
    Returns the Matplotlib figure object.
    """
    
    portfolio_value = portfolio_cumulative * initial_capital
    benchmark_value = benchmark_cumulative * initial_capital
    
    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot portfolio value
    ax.plot(portfolio_value.index, portfolio_value, label='My Portfolio', color='blue', linewidth=2)
    
    # Plot benchmark value
    ax.plot(benchmark_value.index, benchmark_value, label=f'Benchmark ({benchmark_ticker})', color='grey', linestyle='--')
    
    ax.set_title(f'Portfolio Performance vs. {benchmark_ticker}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value (₹)')
    ax.legend()
    ax.grid(True)
    
    # Format Y-axis to show "₹"
    from matplotlib.ticker import FuncFormatter
    formatter = FuncFormatter(lambda x, p: f'₹{x:,.0f}')
    ax.yaxis.set_major_formatter(formatter)
    
    plt.tight_layout()
    
    # Return the figure object
    return fig

def parse_portfolio_weights(text_input):
    """Parses the text area input into a dictionary of weights."""
    weights = {}
    try:
        lines = text_input.strip().split('\n')
        for line in lines:
            if ':' in line:
                ticker, weight = line.split(':')
                ticker = ticker.strip().upper()
                weight = float(weight.strip())
                if weight < 0:
                     st.sidebar.error(f"Weight for {ticker} cannot be negative.")
                     return None
                weights[ticker] = weight
        return weights
    except Exception as e:
        st.sidebar.error(f"Error parsing portfolio: {e}")
        st.sidebar.error("Please use format: TICKER: WEIGHT (e.g., RELIANCE.NS: 0.15)")
        return None

# --- Streamlit App UI ---

st.set_page_config(layout="wide")
st.title("Virtual Investment Portfolio Tracker")
st.write("A Streamlit app to track and analyze your hypothetical portfolio based on the project components.")

# --- Sidebar for Configuration ---
st.sidebar.header("1. Investor & Portfolio Setup")

# Component 1: Investor Profile
st.sidebar.subheader("Component 1: Investor Profile")
initial_capital = st.sidebar.number_input("Initial Capital (₹)", min_value=1_000, value=1_000_000, step=10_000)

# Component 2 & 3: Asset Allocation & Security Selection
st.sidebar.subheader("Components 2 & 3: Assets")
portfolio_example = (
    "RELIANCE.NS: 0.15\n"
    "HDFCBANK.NS: 0.15\n"
    "TATAMOTORS.NS: 0.10\n"
    "INFY.NS: 0.10\n"
    "GOLDBEES.NS: 0.20\n"
    "LIQUIDBEES.NS: 0.30"
)
portfolio_weights_str = st.sidebar.text_area(
    "Portfolio (Ticker: Weight)", 
    value=portfolio_example, 
    height=200,
    help="Enter one asset per line in the format: TICKER: WEIGHT. Use Yahoo Finance tickers (e.g., '.NS' for NSE). Weights should sum to 1.0."
)

st.sidebar.subheader("Component 4: Tracking Setup")
benchmark_ticker = st.sidebar.text_input("Benchmark Ticker", value='^NSEI')

# Default 1 year of data for robust Beta/Sharpe calculation
default_start = date.today().replace(year=date.today().year - 1)
start_date = st.sidebar.date_input("Start Date", value=default_start)
end_date = st.sidebar.date_input("End Date", value=date.today())

risk_free_rate = st.sidebar.slider(
    "Risk-Free Rate (Annual)", 
    min_value=0.0, 
    max_value=0.15, 
    value=0.06, 
    step=0.005, 
    format="%.3f"
)

# --- Main App Body ---

if st.sidebar.button("Run Analysis"):
    
    # 1. Parse and Validate Inputs
    portfolio_weights = parse_portfolio_weights(portfolio_weights_str)
    
    if portfolio_weights is None:
        st.stop()
        
    total_weight = sum(portfolio_weights.values())
    if not np.isclose(total_weight, 1.0):
        st.sidebar.warning(f"Weights sum to {total_weight:.2f}, not 1.0. Results will be scaled.")
        # Optionally, you could stop:
        # st.sidebar.error(f"Weights must sum to 1.0. Current sum is {total_weight:.2f}")
        # st.stop()

    if start_date >= end_date:
        st.sidebar.error("End Date must be after Start Date.")
        st.stop()

    with st.spinner("Analyzing your portfolio..."):
        
        # 2. Get Tickers
        all_tickers = list(portfolio_weights.keys()) + [benchmark_ticker]
        
        # 3. Fetch Data
        price_data = get_stock_data(all_tickers, start_date, end_date)
        
        if price_data is None or price_data.empty:
            st.error("Could not retrieve data for the given tickers or date range.")
            st.stop()
        
        # Handle cases where some tickers failed to download
        missing_tickers = [t for t in all_tickers if t not in price_data.columns]
        if any(missing_tickers):
            st.warning(f"Could not fetch data for: {', '.join(missing_tickers)}. Proceeding without them.")
            # Remove missing tickers from portfolio
            if benchmark_ticker in missing_tickers:
                 st.error(f"Failed to download Benchmark data ({benchmark_ticker}). Cannot proceed.")
                 st.stop()
            portfolio_weights = {t: w for t, w in portfolio_weights.items() if t not in missing_tickers}
            
        # 4. Calculate Performance
        port_returns, bench_returns, port_cum_returns, bench_cum_returns = \
            calculate_portfolio_performance(price_data, portfolio_weights, benchmark_ticker)
        
        # 5. Calculate Metrics
        ann_return, ann_vol, sharpe, beta = \
            calculate_performance_metrics(port_returns, bench_returns, risk_free_rate)
        
        # 6. Get Final Portfolio Value
        final_portfolio_value = port_cum_returns.iloc[-1] * initial_capital
        total_return_pct = (final_portfolio_value / initial_capital) - 1
        
        final_benchmark_value = bench_cum_returns.iloc[-1] * initial_capital
        total_benchmark_return_pct = (final_benchmark_value / initial_capital) - 1
        
        # --- Display Results ---
        
        st.header("Component 4: Performance Tracking")
        st.write(f"Showing results from **{start_date.strftime('%Y-%m-%d')}** to **{end_date.strftime('%Y-%m-%d')}**.")

        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final Portfolio Value", f"₹{final_portfolio_value:,.2f}")
        col2.metric("Total Return", f"{total_return_pct:.2%}")
        col3.metric("Final Benchmark Value", f"₹{final_benchmark_value:,.2f}")
        col4.metric(f"{benchmark_ticker} Return", f"{total_benchmark_return_pct:.2%}")

        st.divider()

        # Plot
        st.subheader("Portfolio Value vs. Benchmark")
        fig = plot_performance(port_cum_returns, bench_cum_returns, initial_capital, benchmark_ticker)
        st.pyplot(fig)
        
        st.divider()
        
        # Metrics Table
        st.subheader("Key Performance Metrics (Annualized)")
        metrics_data = {
            'Metric': [
                'Annualized Return', 
                'Annualized Volatility (Risk)', 
                'Sharpe Ratio (Risk-Adjusted Return)', 
                f'Beta (vs. {benchmark_ticker})'
            ],
            'Value': [
                f"{ann_return:.2%}",
                f"{ann_vol:.2%}",
                f"{sharpe:.2f}",
                f"{beta:.2f}"
            ]
        }
        metrics_df = pd.DataFrame(metrics_data)
        st.table(metrics_df.set_index('Metric'))
        
        # Component 5 & 6: Review & Conclusion
        st.header("Components 5 & 6: Review & Conclusion")
        st.info(
            "Use the data above for your portfolio review and to draw insights.\n\n"
            "**For Review:** Does your Beta match your risk profile? Is your Sharpe Ratio positive? "
            "How did your individual security selections (in the raw data tab) perform?\n\n"
            "**For Rebalancing:** Check the 'Current Holdings' tab to see if any asset has "
            "drifted significantly from its target weight."
        )

        # Data Tabs
        tab1, tab2, tab3 = st.tabs(["Current Holdings", "Daily Portfolio Value", "Raw Price Data"])
        
        with tab1:
            st.subheader("Current Portfolio Holdings")
            # Get last known price for each asset
            last_prices = price_data.drop(columns=[benchmark_ticker], errors='ignore').iloc[-1]
            # Calculate initial allocation in shares
            initial_allocation_value = {t: initial_capital * w for t, w in portfolio_weights.items()}
            # Need first valid price to calculate shares
            first_prices = price_data.drop(columns=[benchmark_ticker], errors='ignore').bfill().iloc[0]
            
            shares = {}
            for t in portfolio_weights.keys():
                if first_prices[t] > 0:
                    shares[t] = initial_allocation_value[t] / first_prices[t]
                else:
                    shares[t] = 0
            
            holdings_df = pd.DataFrame.from_dict(shares, orient='index', columns=['Shares'])
            holdings_df['Current Price'] = last_prices.reindex(holdings_df.index)
            holdings_df['Current Value (₹)'] = holdings_df['Shares'] * holdings_df['Current Price']
            holdings_df['Target Weight'] = pd.Series(portfolio_weights)
            holdings_df['Current Weight'] = holdings_df['Current Value (₹)'] / holdings_df['Current Value (₹)'].sum()
            
            # Format for display
            holdings_df['Target Weight'] = holdings_df['Target Weight'].map('{:,.2%}'.format)
            holdings_df['Current Weight'] = holdings_df['Current Weight'].map('{:,.2%}'.format)
            
            st.dataframe(holdings_df)

        with tab2:
            st.subheader("Daily Portfolio & Benchmark Value")
            value_df = pd.DataFrame({
                'Portfolio Value (₹)': port_cum_returns * initial_capital,
                'Benchmark Value (₹)': bench_cum_returns * initial_capital
            })
            st.dataframe(value_df)

        with tab3:
            st.subheader("Raw 'Adjusted Close' Price Data")
            st.dataframe(price_data)

else:
    st.info("Configure your portfolio in the sidebar and click 'Run Analysis' to begin.")
    st.markdown(
        "**This app helps you with:**\n"
        "- **Component 1-3:** Defining your profile, capital, and asset mix in the sidebar.\n"
        "- **Component 4:** Automatically calculating returns, beta, Sharpe ratio, and volatility.\n"
        "- **Component 5:** Providing a 'Current Holdings' tab to help you identify rebalancing needs.\n"
        "- **Component 6:** Giving you all the data and charts needed to draw your conclusions."
    )
