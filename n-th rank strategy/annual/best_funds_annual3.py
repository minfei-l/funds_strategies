import pandas as pd
import os
import matplotlib.pyplot as plt

mutual_fund_files = [
    "040001.xlsx", "050001.xlsx", "070002.xlsx", "110011.xlsx", "161005.xlsx",
    "163402.xlsx", "202002.xlsx", "270006.xlsx", "377010.xlsx", "260116.xlsx"
]

start_capital = 100
ranks_to_follow = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Load daily return from '日增长率'
def get_daily_return(df):
    df = df[['净值日期', '日增长率']].copy()
    df.columns = ['date', 'daily_return']
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= '2012-01-01') & (df['date'] <= '2025-01-01')]
    df = df.sort_values('date')
    df['daily_return'] = pd.to_numeric(df['daily_return'], errors='coerce') / 100
    return df.dropna()

# Load all funds
stock_returns = {}
for filename in mutual_fund_files:
    fund_code = filename.split('.')[0]
    df = pd.read_excel(filename)
    stock_returns[fund_code] = get_daily_return(df)

# Compute annual return per fund per year
annual_returns = {}
for code, df in stock_returns.items():
    df['year'] = df['date'].dt.year
    for year, group in df.groupby('year'):
        annual_ret = (1 + group['daily_return']).prod() - 1
        annual_returns.setdefault(year, {})[code] = annual_ret

# Simulate "follow N-th best fund from LAST YEAR" strategies
rank_capitals = {rank: start_capital for rank in ranks_to_follow}
rank_history = {rank: {} for rank in ranks_to_follow}

years_sorted = sorted(annual_returns.keys())
init_year = years_sorted[1]  # simulation starts at 2nd year (we use year1 to choose)

for rank in ranks_to_follow:
    rank_history[rank][init_year - 1] = {
        'fund': None,
        'return': 0.0,
        'capital': start_capital
    }

for i in range(1, len(years_sorted)):
    prev_year = years_sorted[i - 1]
    curr_year = years_sorted[i]

    prev_returns = annual_returns[prev_year]
    curr_returns = annual_returns[curr_year]

    if len(prev_returns) < max(ranks_to_follow) or len(curr_returns) == 0:
        continue

    # deterministic tie-break by fund code
    ranked = sorted(prev_returns.items(), key=lambda x: (-x[1], x[0]))

    for rank in ranks_to_follow:
        fund_code, _ = ranked[rank - 1]
        fund_ret = curr_returns.get(fund_code, None)
        if fund_ret is None:
            rank_history[rank][curr_year] = {
                'fund': f'{fund_code} (no data; cash)',
                'return': 0.0,
                'capital': round(rank_capitals[rank], 2)
            }
            continue

        capital = rank_capitals[rank]
        capital *= (1 + fund_ret)
        rank_capitals[rank] = capital
        rank_history[rank][curr_year] = {
            'fund': fund_code,
            'return': round(fund_ret * 100, 2),
            'capital': round(capital, 2)
        }

# Equal-weight of all funds each year
bench_cap = start_capital
bench_hist = {init_year - 1: {'fund': 'All-funds EW', 'return': 0.0, 'capital': start_capital}}

for i in range(1, len(years_sorted)):
    curr_year = years_sorted[i]
    curr_returns = annual_returns.get(curr_year, {})
    if not curr_returns:
        bench_hist[curr_year] = {'fund': 'cash', 'return': 0.0, 'capital': round(bench_cap, 2)}
        continue

    ew_ret = sum(curr_returns.values()) / len(curr_returns)
    bench_cap *= (1 + ew_ret)
    bench_hist[curr_year] = {
        'fund': 'All-funds EW',
        'return': round(ew_ret * 100, 2),
        'capital': round(bench_cap, 2)
    }

# Shanghai Composite Index buy and hold
# Load index (daily % change), compute yearly returns and compound
shanghai_df = pd.read_csv('Shanghai Composite Historical Data.csv')
shanghai_df['Date'] = pd.to_datetime(shanghai_df['Date'], format='%m/%d/%Y')
shanghai_df = shanghai_df.sort_values(by='Date')
shanghai_df['Change %'] = pd.to_numeric(shanghai_df['Change %'].str.rstrip('%'), errors='coerce')
shanghai_df = shanghai_df.dropna(subset=['Change %'])
shanghai_df = shanghai_df[(shanghai_df['Date'] >= '2012-01-01') & (shanghai_df['Date'] <= '2025-01-01')]

shanghai_df['year'] = shanghai_df['Date'].dt.year
sh_yearly_ret = shanghai_df.groupby('year')['Change %'].apply(lambda s: (1 + (s/100)).prod() - 1).to_dict()

sh_cap = start_capital
sh_hist = {init_year - 1: {'fund': 'Shanghai Comp B&H', 'return': 0.0, 'capital': start_capital}}

# Compound across calendar years
for i in range(1, len(years_sorted)):
    yr = years_sorted[i]
    r = sh_yearly_ret.get(yr, None)
    if r is None:
        sh_hist[yr] = {'fund': 'Shanghai Comp B&H', 'return': 0.0, 'capital': round(sh_cap, 2)}
        continue
    sh_cap *= (1 + r)
    sh_hist[yr] = {'fund': 'Shanghai Comp B&H', 'return': round(r * 100, 2), 'capital': round(sh_cap, 2)}

# Rankings: 10 funds only, and 10 funds + Shanghai
fund_codes = [f.split('.')[0] for f in mutual_fund_files]

# Annual return table for funds
df_fund_returns = pd.DataFrame.from_dict(annual_returns, orient='index')
df_fund_returns.index.name = "Year"
df_fund_returns = df_fund_returns.reindex(columns=fund_codes)  # preserve your fund order

# Add Shanghai as another column
sh_series = pd.Series(sh_yearly_ret, name="SHCOMP")
sh_series.index.name = "Year"
df_all_returns = df_fund_returns.copy()
df_all_returns["SHCOMP"] = sh_series

def rank_table_by_row(df_returns: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df_returns.index, columns=df_returns.columns, dtype="float")

    for year, row in df_returns.iterrows():
        items = [(col, row[col]) for col in df_returns.columns if pd.notna(row[col])]
        items_sorted = sorted(items, key=lambda x: (-x[1], x[0]))  # high return first, then name

        ranks = {col: (i + 1) for i, (col, _) in enumerate(items_sorted)}
        for col in df_returns.columns:
            out.loc[year, col] = ranks.get(col, float("nan"))

    return out.astype("Int64")

df_fund_rank = rank_table_by_row(df_fund_returns)   # ranks among only available funds that year
df_all_rank = rank_table_by_row(df_all_returns)     # ranks among available funds + SHCOMP that year

# rows = fund code (left vertical), columns = year (top horizontal)
df_fund_returns_T = df_fund_returns.T
df_fund_rank_T = df_fund_rank.T

df_all_returns_T = df_all_returns.T
df_all_rank_T = df_all_rank.T

# Print rankings
print("\n Annual Return Ranking (10 Funds Only)")
print("Rows = Fund Code | Columns = Year | 1 = Best\n")
print(df_fund_rank_T)

print("\n Annual Return Ranking (10 Funds + Shanghai Composite)")
print("Rows = Fund Code/SHCOMP | Columns = Year | 1 = Best\n")
print(df_all_rank_T)

# Save ranking + return tables
df_fund_returns_T.to_csv("annual_returns_10_funds_T.csv")
df_fund_rank_T.to_csv("annual_rank_10_funds_T.csv", na_rep="")

df_all_returns_T.to_csv("annual_returns_11_including_shcomp_T.csv")
df_all_rank_T.to_csv("annual_rank_11_including_shcomp_T.csv", na_rep="")

print("\n Saved TRANSPOSED tables (rows=fund code, cols=year):")
print("  - annual_returns_10_funds_T.csv")
print("  - annual_rank_10_funds_T.csv")
print("  - annual_returns_11_including_shcomp_T.csv")
print("  - annual_rank_11_including_shcomp_T.csv")

# 3-year total return ranking tables

def rolling_total_return(df_returns: pd.DataFrame, window: int = 3, min_periods: int = 3) -> pd.DataFrame:
    """
    Compute rolling total return over 'window' years for each column.
    Output index is the END year of the window (e.g., 2014 represents 2012-2014).
    """
    df = df_returns.sort_index()
    # rolling product of (1+r), then minus 1
    total = (1 + df).rolling(window=window, min_periods=min_periods).apply(
        lambda x: x.prod(), raw=True
    ) - 1
    return total

def rank_table_by_row(df_returns: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df_returns.index, columns=df_returns.columns, dtype="float")

    for idx, row in df_returns.iterrows():
        items = [(col, row[col]) for col in df_returns.columns if pd.notna(row[col])]
        items_sorted = sorted(items, key=lambda x: (-x[1], x[0]))
        ranks = {col: i + 1 for i, (col, _) in enumerate(items_sorted)}
        for col in df_returns.columns:
            out.loc[idx, col] = ranks.get(col, float("nan"))

    return out.astype("Int64")

# Compute rolling 3-year TOTAL returns
fund_3y_ret = rolling_total_return(df_fund_returns, window=3, min_periods=3)
all_3y_ret  = rolling_total_return(df_all_returns,  window=3, min_periods=3)

# Label the window as "start-end" instead of just end-year, e.g. "2012-2014"
def label_3y_windows(df_3y: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    df_3y = df_3y.copy()
    df_3y.index = [f"{int(y - window + 1)}-{int(y)}" for y in df_3y.index]
    df_3y.index.name = "Window"
    return df_3y

fund_3y_ret = label_3y_windows(fund_3y_ret, window=3)
all_3y_ret  = label_3y_windows(all_3y_ret,  window=3)

# Rank within each 3-year window
fund_3y_rank = rank_table_by_row(fund_3y_ret)
all_3y_rank  = rank_table_by_row(all_3y_ret)

# Transpose so rows = fund code, columns = 3-year window
fund_3y_ret_T  = fund_3y_ret.T
fund_3y_rank_T = fund_3y_rank.T

all_3y_ret_T   = all_3y_ret.T
all_3y_rank_T  = all_3y_rank.T

print("\n 3-Year TOTAL Return Ranking (10 Funds Only) — 1 = Best")
print("Rows = Fund Code | Columns = 3-Year Window\n")
print(fund_3y_rank_T)

print("\n3-Year TOTAL Return Ranking (10 Funds + Shanghai Composite) — 1 = Best")
print("Rows = Fund Code/SHCOMP | Columns = 3-Year Window\n")
print(all_3y_rank_T)

fund_3y_ret_T.to_csv("rolling3y_total_returns_10_funds_T.csv")
fund_3y_rank_T.to_csv("rolling3y_total_rank_10_funds_T.csv", na_rep="")

all_3y_ret_T.to_csv("rolling3y_total_returns_11_including_shcomp_T.csv")
all_3y_rank_T.to_csv("rolling3y_total_rank_11_including_shcomp_T.csv", na_rep="")

print("\n Saved 3-year rolling total return + rank tables:")
print("  - rolling3y_total_returns_10_funds_T.csv")
print("  - rolling3y_total_rank_10_funds_T.csv")
print("  - rolling3y_total_returns_11_including_shcomp_T.csv")
print("  - rolling3y_total_rank_11_including_shcomp_T.csv")

# Display and plot results
print("\n Strategy Results (Follow Top-N Fund Each Year):\n")
for rank in ranks_to_follow:
    print(f" Rank-{rank} Strategy:")
    for year in sorted(rank_history[rank].keys()):
        info = rank_history[rank][year]
        print(f"  {year}: Fund {info['fund']} | Return: {info['return']}% | Capital: ${info['capital']}")
    print(f"  ➤ Final Capital: ${rank_capitals[rank]:.2f}\n")

print(f" Benchmark (All-funds EW) final capital: ${bench_cap:.2f}")
print(f" Shanghai Composite B&H final capital: ${sh_cap:.2f}")

plt.figure(figsize=(10, 6))

# Existing per-rank lines
for rank in ranks_to_follow:
    years = sorted(rank_history[rank].keys())
    capital = [rank_history[rank][y]['capital'] for y in years]
    final_cap = rank_capitals[rank]
    plt.plot(years, capital, marker='o', label=f'Rank-{rank} (${int(final_cap)})')

# Plot EW benchmark
years = sorted(bench_hist.keys())
plt.plot(years, [bench_hist[y]['capital'] for y in years],
         linestyle='--', marker='o', label=f'All-funds EW (${int(bench_cap)})')

# Plot Shanghai Composite B&H
years = sorted(sh_hist.keys())
plt.plot(years, [sh_hist[y]['capital'] for y in years],
         linestyle=':', marker='o', label=f'Shanghai B&H (${int(sh_cap)})')

plt.title("Capital Growth by Following Last-Year Rankings\n+ Benchmarks: All-funds EW & Shanghai Composite B&H")
plt.xlabel("Year")
plt.ylabel("Capital ($)")
plt.legend(title="Strategy (Final Capital)")
plt.grid(True)
plt.tight_layout()
plt.show()


def calculate_max_drawdown(series):
    cumulative = (1 + series).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown

# Per-Year Volatility, Sharpe Ratio, Max Drawdown
print(" Per-Year Metrics (Volatility, Sharpe, Max Drawdown):\n")
results = []

for rank in ranks_to_follow:
    print(f" Rank-{rank} Strategy:")

    for year in sorted(rank_history[rank].keys()):
        fund_code = rank_history[rank][year]['fund']
        if fund_code is None or '(no data; cash)' in str(fund_code) or fund_code == 'cash':
            continue

        df = stock_returns[fund_code]
        df_year = df[df['date'].dt.year == year]
        if df_year.empty or len(df_year) < 30:
            print(f"  {year}: Fund {fund_code} | Insufficient data")
            results.append({
                'Year': year,
                'Rank': rank,
                'Fund': fund_code,
                'Return (%)': None,
                'Volatility (%)': None,
                'Sharpe Ratio': None,
                'Max Drawdown (%)': None
            })
            continue

        daily_returns = df_year['daily_return']
        annual_return = (1 + daily_returns).prod() - 1
        volatility = daily_returns.std(ddof=1) * (252 ** 0.5)
        sharpe = annual_return / volatility if volatility > 0 else float('nan')
        max_dd = calculate_max_drawdown(daily_returns)

        print(f"  {year}: Fund {fund_code} | Return: {annual_return * 100:.2f}% | "
              f"Volatility: {volatility * 100:.2f}% | Sharpe: {sharpe:.2f} | Max Drawdown: {max_dd * 100:.2f}%")

        results.append({
            'Year': year,
            'Rank': rank,
            'Fund': fund_code,
            'Return (%)': round(annual_return * 100, 4),
            'Volatility (%)': round(volatility * 100, 4),
            'Sharpe Ratio': round(sharpe, 4),
            'Max Drawdown (%)': round(max_dd * 100, 4)
        })

    print()

# Save All Metrics
df_results = pd.DataFrame(results).sort_values(by=['Year', 'Rank'])

df_results[['Year', 'Rank', 'Fund', 'Return (%)']].to_csv("annual_growth.csv", index=False)
df_results[['Year', 'Rank', 'Fund', 'Volatility (%)']].to_csv("annual_volatility.csv", index=False)
df_results[['Year', 'Rank', 'Fund', 'Sharpe Ratio']].to_csv("annual_sharpe_ratio.csv", index=False)
df_results[['Year', 'Rank', 'Fund', 'Max Drawdown (%)']].to_csv("annual_max_drawdown.csv", index=False)

print(" Saved to:")
print("  - 'annual_growth.csv'")
print("  - 'annual_volatility.csv'")
print("  - 'annual_sharpe_ratio.csv'")
print("  - 'annual_max_drawdown.csv'")
