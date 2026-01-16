import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Load Shanghai Composite Data
shanghai_df = pd.read_csv('Shanghai Composite Historical Data.csv')
shanghai_df['Date'] = pd.to_datetime(shanghai_df['Date'], format='%m/%d/%Y')
shanghai_df = shanghai_df.sort_values(by='Date')
shanghai_df['Change %'] = shanghai_df['Change %'].str.rstrip('%').astype(float)
shanghai_df['Cumulative Return Multiplier'] = (shanghai_df['Change %'] / 100 + 1).cumprod()

# Load Mutual Fund Data
mutual_fund_files = [
    "040001.xlsx", "050001.xlsx", "070002.xlsx", "110011.xlsx", "161005.xlsx",
    "163402.xlsx", "202002.xlsx", "260116.xlsx", "270006.xlsx", "377010.xlsx"
]

mutual_fund_investment_values = {}

for fund_file in mutual_fund_files:
    code = fund_file[:-5]
    if os.path.exists(fund_file):
        fund_df = pd.read_excel(fund_file)
        fund_df['净值日期'] = pd.to_datetime(fund_df['净值日期'], format='%Y-%m-%d', errors='coerce')
        fund_df = fund_df.sort_values(by='净值日期')
        fund_df['日增长率'] = pd.to_numeric(fund_df['日增长率'], errors='coerce') / 100
        fund_df = fund_df.dropna(subset=['日增长率'])
        fund_df['Cumulative Return Multiplier'] = (fund_df['日增长率'] + 1).cumprod()
        mutual_fund_investment_values[code] = fund_df[['净值日期', 'Cumulative Return Multiplier']].set_index(
            '净值日期')

# Load Strategy Fund Data
allocation_df = pd.read_csv('formals.csv')
allocation_df['code'] = allocation_df['code'].astype(str).str.zfill(6)
for stock in allocation_df.columns[2:]:
    allocation_df[stock] = allocation_df[stock].replace('%', '', regex=True).astype(float) / 100

# Load Sector Data
sector_name_map = {
    "Consumer Staples": "Cons Staples",
    "Energy": "Energy",
    "Financials": "Financials",
    "Health Care": "Health Care",
    "Industrial": "Industrials",
    "Information Technology": "Info technology",
    "Materials": "Materials",
    "Telecommunication Services": "Communication Services",
    "Utility": "Utilities"
}

sector_data = {}
if os.path.exists('sectors.xlsx'):
    sect_df = pd.read_excel('sectors.xlsx', header=0)
    sect_df['Date'] = pd.to_datetime(sect_df['Date'], errors='coerce')
    sect_df = sect_df.dropna(subset=['Date']).sort_values('Date').set_index('Date')
    for col in sect_df.columns:
        sect_df[col] = pd.to_numeric(sect_df[col], errors='coerce')
    sect_df = sect_df / 100.0
    for orig_name, sheet_name in sector_name_map.items():
        if sheet_name in sect_df.columns:
            s = sect_df[sheet_name].fillna(0.0)
            cm = (s + 1.0).cumprod()
            sector_data[orig_name] = cm

# Calculate Strategy Fund Values
strategy_fund_values = {}
for code in allocation_df['code'].unique():
    portfolio_series = pd.Series(dtype=float)
    last_val = 100.0

    for year in range(2013, 2025):
        allocations = allocation_df[(allocation_df['code'] == code) & (allocation_df['Year'] == year - 1)]
        if allocations.empty:
            continue

        alloc_row = allocations.iloc[0]
        candidate_names = list(sector_name_map.keys())
        weights = {
            stock: float(alloc_row.get(stock, np.nan))
            for stock in candidate_names
            if pd.notna(alloc_row.get(stock, np.nan)) and float(alloc_row.get(stock, np.nan)) > 0
        }
        if not weights:
            continue

        wsum = sum(weights.values())
        if wsum <= 0:
            continue
        weights = {k: v / wsum for k, v in weights.items()}

        year_cols = {}
        for stock, w in weights.items():
            s = sector_data.get(stock, None)
            if s is None or s.empty:
                continue
            s_y = s[s.index.year == year]
            if s_y.empty:
                continue
            s_y = s_y / s_y.iloc[0]
            year_cols[stock] = s_y

        if not year_cols:
            continue

        df_year = pd.DataFrame(year_cols).dropna(how='any')
        if df_year.empty:
            continue

        port_year = (df_year.mul(pd.Series(weights))).sum(axis=1) * last_val

        if portfolio_series.empty:
            portfolio_series = port_year.copy()
        else:
            if portfolio_series.index[-1] == port_year.index[0]:
                port_year = port_year.iloc[1:]
            portfolio_series = pd.concat([portfolio_series, port_year])

        last_val = port_year.iloc[-1]

    if not portfolio_series.empty:
        strategy_fund_values[code] = portfolio_series

# Normalize All Data to Start at $100
start_date = pd.Timestamp('2013-01-01')
end_date = pd.Timestamp('2025-01-01')

# Shanghai Composite
shanghai_df = shanghai_df[shanghai_df['Date'] >= start_date].copy()
shanghai_start_value = shanghai_df['Cumulative Return Multiplier'].iloc[0]
shanghai_df['Investment Value'] = (shanghai_df['Cumulative Return Multiplier'] / shanghai_start_value) * 100

# Mutual Funds
for code, fund_df in mutual_fund_investment_values.items():
    fund_df = fund_df[fund_df.index >= start_date].copy()
    if not fund_df.empty:
        fund_start_value = fund_df['Cumulative Return Multiplier'].iloc[0]
        fund_df['Investment Value'] = (fund_df['Cumulative Return Multiplier'] / fund_start_value) * 100
    mutual_fund_investment_values[code] = fund_df

# Strategy Funds
for code, strategy_values in strategy_fund_values.items():
    strategy_values = strategy_values[strategy_values.index >= start_date].copy()
    if not strategy_values.empty:
        strategy_start_value = strategy_values.iloc[0]
        strategy_fund_values[code] = (strategy_values / strategy_start_value) * 100

# Combine All Capital Values into One File
all_capital_data = []

# Add Shanghai Composite data
shanghai_capital = shanghai_df[['Date', 'Investment Value']].copy()
shanghai_capital['Fund Code'] = 'Shanghai Composite'
shanghai_capital['Fund Type'] = 'Index'
all_capital_data.append(shanghai_capital.rename(columns={'Date': 'Date', 'Investment Value': 'Capital Value'}))

# Add Mutual Funds data
for code, fund_df in mutual_fund_investment_values.items():
    if not fund_df.empty:
        fund_capital = fund_df.reset_index()[['净值日期', 'Investment Value']].copy()
        fund_capital['Fund Code'] = code
        fund_capital['Fund Type'] = 'Mutual Fund'
        all_capital_data.append(fund_capital.rename(columns={'净值日期': 'Date', 'Investment Value': 'Capital Value'}))

# Add Strategy Funds data
for code, strategy_series in strategy_fund_values.items():
    strategy_capital = strategy_series.reset_index()
    strategy_capital.columns = ['Date', 'Capital Value']
    strategy_capital['Fund Code'] = code
    strategy_capital['Fund Type'] = 'Strategy Fund'
    all_capital_data.append(strategy_capital)

# Combine all data
combined_capital_df = pd.concat(all_capital_data, ignore_index=True)

# Sort by Date, Fund Type, and Fund Code
combined_capital_df = combined_capital_df.sort_values(['Date', 'Fund Type', 'Fund Code'])

# Save to CSV
combined_capital_df.to_csv('all_capital_values.csv', index=False)
print("All capital values saved to 'all_capital_values.csv'")

# --- Plot Graphs ---
final_records = []
matched_codes = [code for code in mutual_fund_investment_values if code in strategy_fund_values]

for code in matched_codes:
    plt.figure(figsize=(12, 6))

    shanghai_df_filtered = shanghai_df[shanghai_df['Date'] >= start_date].copy()

    fund_df = mutual_fund_investment_values[code]
    fund_df_filtered = fund_df[fund_df.index >= start_date].copy()
    if fund_df_filtered.empty:
        plt.close()
        continue

    strategy_series = strategy_fund_values[code]
    strategy_series_filtered = strategy_series[strategy_series.index >= start_date].copy()
    if strategy_series_filtered.empty:
        plt.close()
        continue

    sh_series = shanghai_df_filtered.set_index('Date')['Investment Value']
    mf_series = fund_df_filtered['Investment Value']
    sf_series = strategy_series_filtered

    common_idx = sh_series.index.intersection(mf_series.index).intersection(sf_series.index)
    common_idx = common_idx[(common_idx >= start_date) & (common_idx <= end_date)]
    if common_idx.empty:
        plt.close()
        continue
    final_date = common_idx.max()

    final_shanghai = float(sh_series.loc[final_date])
    final_fund = float(mf_series.loc[final_date])
    final_strategy = float(sf_series.loc[final_date])

    final_records.append({
        "Code": code,
        "Final Date": final_date.date(),
        "Final Fund": final_fund,
        "Final Strategy": final_strategy
    })

    plt.plot(sh_series.loc[common_idx].index, sh_series.loc[common_idx].values,
             label=f'Shanghai Composite (${int(round(final_shanghai))})', color='blue')
    plt.plot(mf_series.loc[common_idx].index, mf_series.loc[common_idx].values,
             label=f'Mutual Fund {code} (${int(round(final_fund))})', color='red')
    plt.plot(sf_series.loc[common_idx].index, sf_series.loc[common_idx].values,
             label=f'Strategy Fund {code} (${int(round(final_strategy))})', color='green')

    plt.title(f"Investment Comparison: {code}")
    plt.xlabel("Date")
    plt.ylabel("Investment Value ($)")
    plt.legend()
    plt.grid(True)
    plt.xlim(start_date, end_date)
    plt.tight_layout()
    plt.show()

# Build a tidy table of final values
final_df = pd.DataFrame(final_records)
final_df["Final Fund (rounded)"] = final_df["Final Fund"].round(0).astype(int)
final_df["Final Strategy (rounded)"] = final_df["Final Strategy"].round(0).astype(int)
final_df["Fund Capital"] = (final_df["Final Fund"] - 100).round(0).astype(int)
final_df["Strategy Capital"] = (final_df["Final Strategy"] - 100).round(0).astype(int)
final_df = final_df.sort_values("Code")

print("\nFinal values at shared end date per code (base=100 at 2013-01-01):\n")
print(final_df[[
    "Code", "Final Date",
    "Final Fund (rounded)", "Final Strategy (rounded)",
    "Fund Capital", "Strategy Capital"
]].to_string(index=False))


# Functions to calculate Volatility, Max Drawdown, and Sharpe Ratio
def calculate_volatility(series, freq=252):
    daily_returns = series.pct_change().dropna()
    return np.std(daily_returns) * np.sqrt(freq) * 100


def calculate_max_drawdown(series):
    cumulative_max = series.cummax()
    drawdowns = (series - cumulative_max) / cumulative_max
    return drawdowns.min() * 100


def calculate_sharpe_ratio(series, freq=252):
    daily_returns = series.pct_change().dropna()
    return (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(freq)


yearly_metrics = []


def calc_metrics_for_period(series):
    if len(series) < 2:
        return np.nan, np.nan, np.nan
    vol = calculate_volatility(series)
    mdd = calculate_max_drawdown(series)
    sharpe = calculate_sharpe_ratio(series)
    return vol, mdd, sharpe


# Process Shanghai Composite
for year in range(start_date.year, end_date.year + 1):
    yearly_data = shanghai_df[shanghai_df['Date'].dt.year == year]['Investment Value']
    vol, mdd, sharpe = calc_metrics_for_period(yearly_data)
    yearly_metrics.append([year, 'Shanghai Composite', 'Shanghai Composite', vol, mdd, sharpe])

# Process Mutual Funds & Strategy Funds
for code in matched_codes:
    # Mutual Fund
    fund_df = mutual_fund_investment_values[code]
    for year in range(start_date.year, end_date.year + 1):
        yearly_data = fund_df[fund_df.index.year == year]['Investment Value']
        vol, mdd, sharpe = calc_metrics_for_period(yearly_data)
        yearly_metrics.append([year, f'Mutual Fund {code}', 'Mutual Fund', vol, mdd, sharpe])

    # Strategy Fund
    strategy_series = strategy_fund_values[code]
    for year in range(start_date.year, end_date.year + 1):
        yearly_data = strategy_series[strategy_series.index.year == year]
        vol, mdd, sharpe = calc_metrics_for_period(yearly_data)
        yearly_metrics.append([year, f'Strategy Fund {code}', 'Strategy Fund', vol, mdd, sharpe])

# Create DataFrame
yearly_metrics_df = pd.DataFrame(yearly_metrics, columns=['Year', 'Fund', 'Category', 'Volatility', 'MDD', 'Sharpe'])
yearly_metrics_df[['Volatility', 'MDD', 'Sharpe']] = yearly_metrics_df[['Volatility', 'MDD', 'Sharpe']].round(2)
yearly_metrics_df = yearly_metrics_df.rename(columns={
    'Volatility': 'Volatility (%)',
    'MDD': 'MDD (%)',
    'Sharpe': 'Sharpe Ratio'
})

yearly_metrics_df.to_csv('annual_fund_metrics.csv', index=False)


# Extract 6-digit codes for file naming
def extract_code_from_fund(name: str):
    s = str(name)
    parts = s.split()
    for p in reversed(parts):
        if p.isdigit() and len(p) == 6:
            return p
    return "index"


# Create separate DataFrames for each metric
volatility_df = yearly_metrics_df.pivot(index='Year', columns='Fund', values='Volatility (%)')
mdd_df = yearly_metrics_df.pivot(index='Year', columns='Fund', values='MDD (%)')
sharpe_df = yearly_metrics_df.pivot(index='Year', columns='Fund', values='Sharpe Ratio')

# Save each metric to separate CSV files
volatility_df.to_csv('annual_volatility_by_fund.csv')
mdd_df.to_csv('annual_mdd_by_fund.csv')
sharpe_df.to_csv('annual_sharpe_ratio_by_fund.csv')

print("Separate metric files saved:")
print("- annual_volatility_by_fund.csv")
print("- annual_mdd_by_fund.csv")
print("- annual_sharpe_ratio_by_fund.csv")

# Load the combined yearly metrics
df = pd.read_csv("annual_fund_metrics.csv")
for c in ["Volatility (%)", "MDD (%)", "Sharpe Ratio"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)


# Extract 6-digit code
def extract_code(name: str):
    s = str(name)
    parts = s.split()
    for p in reversed(parts):
        if p.isdigit() and len(p) == 6:
            return p
    return None


df["Code"] = df["Fund"].apply(extract_code)
df.loc[df["Category"] == "Shanghai Composite", "Code"] = "Index"

# Compute averages
avg_df = (
    df.groupby(["Category", "Code"], dropna=False)
    .agg(
        avg_volatility=("Volatility (%)", "mean"),
        avg_mdd=("MDD (%)", "mean"),
        avg_sharpe=("Sharpe Ratio", "mean"),
        n_years=("Year", "nunique")
    )
    .reset_index()
)

avg_df[["avg_volatility", "avg_mdd", "avg_sharpe"]] = avg_df[["avg_volatility", "avg_mdd", "avg_sharpe"]].round(2)
avg_df = avg_df.sort_values(["Category", "Code"])

avg_df = avg_df.rename(columns={
    'avg_volatility': 'Volatility (%)',
    'avg_mdd': 'MDD (%)',
    'avg_sharpe': 'Sharpe Ratio'
})

avg_strategy = avg_df[avg_df["Category"] == "Strategy Fund"].drop(columns="Category")
avg_mutual = avg_df[avg_df["Category"] == "Mutual Fund"].drop(columns="Category")
avg_index = avg_df[avg_df["Category"] == "Shanghai Composite"]

print("Final Averages:")
print(avg_df.to_string(index=False))
print("\nStrategy Fund averages:\n", avg_strategy.to_string(index=False))
print("\nMutual Fund averages:\n", avg_mutual.to_string(index=False))
print("\nBenchmark averages:\n", avg_index.to_string(index=False))

avg_df.to_csv("avg_metrics_by_code.csv", index=False)