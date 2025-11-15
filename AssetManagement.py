import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import datetime

st.set_page_config(page_title="Asset Management Dashboard", layout="wide")
st.title("Asset Management Dashboard")
st.markdown(
    "An all-in-one financial analysis toolkit for portfolio optimization, mutual fund analysis, trading backtesting, and asset allocation"
)
end_date = datetime.datetime.today()

tab_options = [
    "Stock Price Viewer",
    "Portfolio Optimization",
    "Mutual Fund Analysis",
    "Asset Allocation",
    "Investment Planner",
    "Tax & Inflation Adjusted Returns"
]

selected_tab = st.sidebar.radio("", tab_options)

#Home Page
if selected_tab == "Stock Price Viewer":
    st.header("Stock Price Viewer")

    chart_tickers = st.text_input(
        "Enter Stock Tickers to Plot (comma-separated)",
        "BHEL.NS,IRFC.NS,IOC.NS,HDFCBANK.NS,ITC.NS"
    )
    chart_tickers = [t.strip() for t in chart_tickers.split(",") if t.strip()]

    if chart_tickers:
        data = yf.download(chart_tickers, start="2020-01-01", end=end_date)

        if "Adj Close" in data.columns.get_level_values(-1):
            adj_close = data["Adj Close"]
        else:
            adj_close = data["Close"]

        # Plot
        df_plot = adj_close.reset_index().melt(id_vars="Date", var_name="tickers", value_name="Price")
        fig = px.line(df_plot, x="Date", y="Price", color="tickers", title="Stock Price History")
        fig.update_traces(line=dict(width=2))
        fig.update_layout(xaxis_title="Date", yaxis_title="Price (₹)")
        st.plotly_chart(fig, use_container_width=True)

        # Stock summary
        info_list = []
        for ticker in chart_tickers:
            hist = yf.Ticker(ticker).history(period="1y")
            if not hist.empty:
                current_price = hist['Close'][-1]
                high_52 = hist['High'].max()
                low_52 = hist['Low'].min()
                volume = hist['Volume'][-1]
                open_price = hist['Open'][-1]
                close_price = hist['Close'][-1]

                info_list.append({
                    "Ticker": ticker,
                    "Current Price (₹)": current_price,
                    "52W High (₹)": high_52,
                    "52W Low (₹)": low_52,
                    "Volume": volume,
                    "Open (₹)": open_price,
                    "Close (₹)": close_price
                })

        if info_list:
            df_info = pd.DataFrame(info_list)
            st.subheader("Stock Summary (Current & 52-Week Data)")
            st.dataframe(df_info.style.format({
                "Current Price (₹)": "{:,.2f}",
                "52W High (₹)": "{:,.2f}",
                "52W Low (₹)": "{:,.2f}",
                "Volume": "{:,}",
                "Open (₹)": "{:,.2f}",
                "Close (₹)": "{:,.2f}"
            }))

            st.subheader("Summary Insights")
            for idx, row in df_info.iterrows():
                ticker = row['Ticker']
                price = row['Current Price (₹)']
                high = row['52W High (₹)']
                low = row['52W Low (₹)']
                vol = row['Volume']

                if price <= low + (high - low) * 0.1:
                    trend = "Near 52-week low → Potential buying opportunity"
                elif price >= high - (high - low) * 0.1:
                    trend = "Near 52-week high → Possible resistance / caution"
                else:
                    trend = "Trading in mid-range → Stable trend"

                hist_1y = yf.Ticker(ticker).history(period="1y")['Close']
                perf_pct = ((price / hist_1y.iloc[0]) - 1) * 100
                st.write(f"**{ticker}:** {trend}. Price change over last 1 year: {perf_pct:.2f}% (Volume: {vol:,})")
    else:
        st.warning("Please enter at least one ticker.")

# Portfolio Optimization Tab
elif selected_tab == "Portfolio Optimization":
    st.header("Portfolio Optimization (Stocks)")
    tickers = st.text_input(
        "Enter Stock Tickers (Comma-Separated)",
        "BHEL.NS,IRFC.NS,IOC.NS,HDFCBANK.NS,ITC.NS",
    )
    tickers = [t.strip() for t in tickers.split(",") if t.strip()]

    investment = st.number_input("Enter Investment Amount (₹)", min_value=1000, value=100000)

    if st.button("Run Portfolio Optimization"):
        if len(tickers) == 0:
            st.error("Please enter at least 1 ticker.")
        else:
            data = yf.download(tickers, start="2020-01-01", end=end_date)
            if "Adj Close" in data.columns.get_level_values(-1):
                adj_close = data["Adj Close"]
            else:
                adj_close = data["Close"]

            returns = adj_close.pct_change().dropna()
            mean_returns = returns.mean()
            cov_matrix = returns.cov()

            num_portfolios = 5000
            results = np.zeros((3, num_portfolios))
            weights_record = []

            for i in range(num_portfolios):
                weights = np.random.random(len(tickers))
                weights /= np.sum(weights)
                weights_record.append(weights)

                port_return = np.dot(mean_returns.values, weights) * 252
                port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values * 252, weights)))

                results[0, i] = port_std
                results[1, i] = port_return
                results[2, i] = results[1, i] / results[0, i] if results[0, i] != 0 else 0

            max_sharpe_idx = np.argmax(results[2])
            best_weights = weights_record[max_sharpe_idx]

            best_portfolio = pd.DataFrame({
                "Ticker": tickers,
                "Allocation (%)": np.round(best_weights * 100, 2),
                "Amount (₹)": np.round(best_weights * investment, 2),
                "Closed": adj_close.iloc[-1].values
            }).sort_values(by="Allocation (%)", ascending=False).reset_index(drop=True)

            st.subheader("Optimal Portfolio Allocation (Max Sharpe)")
            st.dataframe(best_portfolio, use_container_width=True)

            best_ret = results[1, max_sharpe_idx]
            best_vol = results[0, max_sharpe_idx]
            best_sharpe = results[2, max_sharpe_idx]
            expected_return_value = investment * best_ret

            st.markdown(f"""
            **Expected Annual Return:** {best_ret:.2%}  
            **Annual Volatility:** {best_vol:.2%}  
            **Sharpe Ratio:** {best_sharpe:.3f}  
            **Expected Annual Profit:** ₹{expected_return_value:,.0f}  
            **Expected Portfolio Value after 1 Year:** ₹{investment + expected_return_value:,.0f}
            """)

# Mutual Fund Analysis Tab
elif selected_tab == "Mutual Fund Analysis":
    st.header("Mutual Fund Risk vs Return Analyzer")

    funds_input = st.text_input("Enter Mutual Fund Codes (comma separated)",
                                "118834,100122,122640,134923,152694,152687,152692,148035,134643")
    funds = [f.strip() for f in funds_input.split(",")]

    if st.button("Analyze Mutual Funds"):
        results = []
        for code in funds:
            url = f"https://api.mfapi.in/mf/{code}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                meta = data.get("meta", {})
                name = meta.get("scheme_name", f"Unknown ({code})")
                nav_data = data.get("data", [])
                df = pd.DataFrame(nav_data)
                if not df.empty and 'nav' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
                    df['nav'] = df['nav'].astype(float)
                    df = df.sort_values('date')
                    df['daily_return'] = df['nav'].pct_change()
                    mean_daily = df['daily_return'].mean()
                    vol_daily = df['daily_return'].std()
                    annual_return = (1 + mean_daily) ** 252 - 1
                    annual_volatility = vol_daily * np.sqrt(252)
                    sharpe = annual_return / annual_volatility if annual_volatility != 0 else np.nan

                    results.append({
                        "Scheme Code": code,
                        "Fund Name": name,
                        "Annual Return": annual_return * 100,
                        "Volatility": annual_volatility * 100,
                        "Sharpe Ratio": sharpe
                    })

        df_results = pd.DataFrame(results)
        if not df_results.empty:
            filtered = df_results[df_results["Annual Return"] > df_results["Volatility"]]
            st.subheader("Filtered Funds (Return > Volatility)")
            st.dataframe(filtered[["Scheme Code", "Fund Name", "Annual Return", "Volatility", "Sharpe Ratio"]]
                         .style.format({"Annual Return": "{:.2f}%", "Volatility": "{:.2f}%", "Sharpe Ratio": "{:.2f}"}))

# Asset Allocation Tab
elif selected_tab == "Asset Allocation":
    st.header("Asset Allocation")
    age = st.number_input("Enter your age:", min_value=18, max_value=100, value=30)
    risk = st.selectbox("Risk Tolerance:", ["Low", "Medium", "High"])

    def allocation(age, risk):
        base_equity = 100 - age
        if risk == "High":
            equity = base_equity + 20
        elif risk == "Medium":
            equity = base_equity
        else:
            equity = base_equity - 20
        equity = np.clip(equity, 0, 100)
        remaining = 100 - equity
        debt = remaining * 0.6
        gold = remaining * 0.3
        cash = remaining * 0.1
        return {'Equity': equity, 'Debt': debt, 'Gold': gold, 'Cash': cash}

    alloc = allocation(age, risk)
    st.subheader("Recommended Asset Allocation")
    st.write(pd.DataFrame(alloc, index=["Allocation %"]).T)

# Investment Planner Tab
elif selected_tab == "Investment Planner":
    st.header("Financial Goal Planner")
    investment_amount = st.number_input("Enter Your Initial Amount (₹)", min_value=10000, value=500000)
    goal_amount = st.number_input("Enter Your Goal Amount (₹)", min_value=10000, value=1000000)
    goal_years = st.slider("Select investment duration (years)", 1, 30, 10)
    expected_return = st.slider("Expected Annual Return (%)", 1, 100, 10)
    volatility = st.slider("Enter expected annual volatility (%)", 1, 15, 10)

    annual_return = expected_return / 100
    monthly_rate = (1 + annual_return) ** (1 / 12) - 1
    months = goal_years * 12

    lump_sum_needed = goal_amount / ((1 + annual_return) ** goal_years)
    required_sip = goal_amount / (((1 + monthly_rate) ** months - 1) / monthly_rate)

    st.markdown("### 💡 Goal Analysis")
    st.markdown(f"""
        **Goal Amount:** ₹{goal_amount:,.0f}  
        **Time Horizon:** {goal_years} years  
        **Expected Annual Return:** {expected_return:.2f}%  

        **To reach your goal:**
        - Lump Sum Investment Required **today:** ₹{lump_sum_needed:,.0f}  
        - OR Monthly SIP Required **for {goal_years} years:** ₹{required_sip:,.0f}  
        """)


# Tax & Inflation Adjusted Returns Tab
elif selected_tab == "Tax & Inflation Adjusted Returns":
    st.header("Tax & Inflation Adjusted Returns + Growth Projection")
    investment_amount = st.number_input("Enter Investment Amount (₹)", min_value=1000, value=100000, key="inv_amt_tab4")
    expected_return = st.number_input("Enter Expected Annual Return (%)", min_value=0.0, value=12.0, step=0.5,
                                      key="exp_ret_tab4")
    years = st.slider("Investment Duration (years)", 1, 30, 10, key="years_tab4")
    tax_rate = st.slider("Select Tax Rate (%)", 0, 50, 15, key="tax_tab4")
    inflation_rate = st.slider("Expected Inflation Rate (%)", 0, 15, 6, key="inf_tab4")

    r = expected_return / 100
    t = tax_rate / 100
    i = inflation_rate / 100

    growth = [investment_amount * ((1 + r) ** y) for y in range(1, years + 1)]
    final_value = growth[-1]
    taxed_value = investment_amount * ((1 + r * (1 - t)) ** years)
    real_return_rate = ((1 + r) / (1 + i)) - 1
    real_final_value = investment_amount * ((1 + real_return_rate) ** years)
    total_gain = final_value - investment_amount
    total_return_pct = (final_value / investment_amount - 1) * 100

    st.subheader("📊 Adjusted Return Summary")
    st.markdown(f"""
        **Initial Investment:** ₹{investment_amount:,.0f}  
        **Nominal Final Value (no tax/inflation):** ₹{final_value:,.0f}  
        **After-Tax Final Value:** ₹{taxed_value:,.0f}  
        **Inflation-Adjusted Final Value:** ₹{real_final_value:,.0f}  

        **Total Gain:** ₹{total_gain:,.0f}  
        **Total Return:** {total_return_pct:.2f}%  
        **Effective Real Annual Return:** {real_return_rate * 100:.2f}%  
        **Investment Period:** {years} years  
        **Expected Annual Return:** {expected_return:.2f}%  

        """)

