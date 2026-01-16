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

# Simulate follow N-th best fund from last year strategies
rank_capitals = {rank: start_capital for rank in ranks_to_follow}
rank_history = {rank: {} for rank in ranks_to_follow}

years_sorted = sorted(annual_returns.keys())
init_year = years_sorted[1]  # simulation starts at 2nd year

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

    ranked = sorted(prev_returns.items(), key=lambda x: x[1], reverse=True)

    for rank in ranks_to_follow:
        fund_code, _ = ranked[rank - 1]
        fund_ret = curr_returns.get(fund_code, None)
        if fund_ret is None:
            continue

        capital = rank_capitals[rank]
        capital *= (1 + fund_ret)
        rank_capitals[rank] = capital
        rank_history[rank][curr_year] = {
            'fund': fund_code,
            'return': round(fund_ret * 100, 2),
            'capital': round(capital, 2)
        }


# Display and Plot results
print("\n📈 Strategy Results (Follow Top-N Fund Each Year):\n")
for rank in ranks_to_follow:
    print(f"▶️ Rank-{rank} Strategy:")
    for year in sorted(rank_history[rank].keys()):
        info = rank_history[rank][year]
        print(f"  {year}: Fund {info['fund']} | Return: {info['return']}% | Capital: ${info['capital']}")
    print(f"  ➤ Final Capital: ${rank_capitals[rank]:.2f}\n")

plt.figure(figsize=(10, 6))
for rank in ranks_to_follow:
    years = sorted(rank_history[rank].keys())
    capital = [rank_history[rank][y]['capital'] for y in years]
    final_cap = rank_capitals[rank]
    plt.plot(years, capital, marker='o', label=f'Rank-{rank} (${int(final_cap)})')
    # plt.text(years[-1], capital[-1], f"${int(capital[-1])}", fontsize=9, verticalalignment='top')

plt.title("Capital Growth by Following Top-N Fund Every Years\n(Based on Past Annual Return)")
plt.xlabel("Year")
plt.ylabel("Capital ($)")
plt.legend(title="Strategy (Final Capital)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Max Drawdown Calculation Function
def calculate_max_drawdown(series):
    cumulative = (1 + series).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown

# Step 6: Per-Year Volatility, Sharpe Ratio, Max Drawdown
print("Per-Year Metrics (Volatility, Sharpe, Max Drawdown):\n")
results = []

for rank in ranks_to_follow:
    print(f"Rank-{rank} Strategy:")

    for year in sorted(rank_history[rank].keys()):
        fund_code = rank_history[rank][year]['fund']
        if fund_code is None:
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
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by=['Year', 'Rank'])


df_results[['Year', 'Rank', 'Fund', 'Return (%)']].to_csv("annual_growth.csv", index=False)
df_results[['Year', 'Rank', 'Fund', 'Volatility (%)']].to_csv("annual_volatility.csv", index=False)
df_results[['Year', 'Rank', 'Fund', 'Sharpe Ratio']].to_csv("annual_sharpe_ratio.csv", index=False)
df_results[['Year', 'Rank', 'Fund', 'Max Drawdown (%)']].to_csv("annual_max_drawdown.csv", index=False)

print("Saved to:")
print("  - 'annual_growth.csv'")
print("  - 'annual_volatility.csv'")
print("  - 'annual_sharpe_ratio.csv'")
print("  - 'annual_max_drawdown.csv'")

# Load the CSVs
sharpe = pd.read_csv('annual_sharpe_ratio.csv')
volatility = pd.read_csv('annual_volatility.csv')
mdd = pd.read_csv('annual_max_drawdown.csv')

print("\n========== SHARPE RATIO ==========")
print("Average Sharpe Ratio by Rank:")
print(sharpe.groupby('Rank')['Sharpe Ratio'].mean().sort_values(ascending=False))
best_sharpe = sharpe.loc[sharpe['Sharpe Ratio'].idxmax()]
print(f"\nBest Sharpe Ratio:\n  Year: {best_sharpe['Year']}, Rank: {best_sharpe['Rank']}, Value: {best_sharpe['Sharpe Ratio']:.3f}")
worst_sharpe = sharpe.loc[sharpe['Sharpe Ratio'].idxmin()]
print(f"Worst Sharpe Ratio:\n  Year: {worst_sharpe['Year']}, Rank: {worst_sharpe['Rank']}, Value: {worst_sharpe['Sharpe Ratio']:.3f}")

print("\n========== VOLATILITY (%) ==========")
print("Average Volatility by Rank (%):")
print(volatility.groupby('Rank')['Volatility (%)'].mean().sort_values())
most_volatile = volatility.loc[volatility['Volatility (%)'].idxmax()]
print(f"\nMost Volatile:\n  Year: {most_volatile['Year']}, Rank: {most_volatile['Rank']}, Value: {most_volatile['Volatility (%)']:.2f}%")
least_volatile = volatility.loc[volatility['Volatility (%)'].idxmin()]
print(f"Least Volatile:\n  Year: {least_volatile['Year']}, Rank: {least_volatile['Rank']}, Value: {least_volatile['Volatility (%)']:.2f}%")

print("\n========== MAX DRAWDOWN (%) ==========")
print("Average Max Drawdown by Rank (%):")
print(mdd.groupby('Rank')['Max Drawdown (%)'].mean().sort_values())
worst_drawdown = mdd.loc[mdd['Max Drawdown (%)'].idxmin()]
print(f"\nWorst Max Drawdown:\n  Year: {worst_drawdown['Year']}, Rank: {worst_drawdown['Rank']}, Value: {worst_drawdown['Max Drawdown (%)']:.2f}%")
best_drawdown = mdd.loc[mdd['Max Drawdown (%)'].idxmax()]
print(f"Smallest Max Drawdown:\n  Year: {best_drawdown['Year']}, Rank: {best_drawdown['Rank']}, Value: {best_drawdown['Max Drawdown (%)']:.2f}%")

# Load growth data if available
try:
    growth = pd.read_csv('annual_growth.csv')
    growth_available = True
except:
    growth_available = False

summary = pd.DataFrame({
    'Sharpe_Mean': sharpe.groupby('Rank')['Sharpe Ratio'].mean(),
    'Sharpe_Std': sharpe.groupby('Rank')['Sharpe Ratio'].std(),
    'Vol_Mean': volatility.groupby('Rank')['Volatility (%)'].mean(),
    'Vol_Std': volatility.groupby('Rank')['Volatility (%)'].std(),
    'MDD_Mean': mdd.groupby('Rank')['Max Drawdown (%)'].mean(),
    'MDD_Std': mdd.groupby('Rank')['Max Drawdown (%)'].std()
})

if growth_available:
    summary['Return_Mean'] = growth.groupby('Rank')['Return (%)'].mean()
    summary['Return_Std'] = growth.groupby('Rank')['Return (%)'].std()

# Rank for low risk (lowest volatility, drawdown), high Sharpe, low std
print("Summary stability/risk table:\n")
print(summary.sort_values(by=['Vol_Mean','MDD_Mean','Sharpe_Mean'])) # Or tweak sort to your focus

# Best overall:
print("\nBest by each metric:")
print("Lowest Volatility:", summary['Vol_Mean'].idxmin())
print("Lowest MDD:", summary['MDD_Mean'].idxmin())
print("Highest Sharpe:", summary['Sharpe_Mean'].idxmax())
if growth_available:
    print("Highest Mean Return:", summary['Return_Mean'].idxmax())
    print("Lowest Std Return:", summary['Return_Std'].idxmin())

# Initial capital
start_capital = 100

# Assuming rank_capitals is a dictionary
cumulative_growth = {rank: (final_capital - start_capital) / start_capital * 100
                     for rank, final_capital in rank_capitals.items()}

# Sort by cumulative growth, descending
top3 = sorted(cumulative_growth.items(), key=lambda x: x[1], reverse=True)[:3]

print("Top 3 strategies by cumulative growth:")
for rank, growth in top3:
    print(f"  Rank-{rank}: {growth:.2f}% cumulative growth (Final capital: ${rank_capitals[rank]:.2f})")


print("\n Yearly Ranking of Strategies by Capital (Descending):\n")

# Collect all years across all ranks
all_years = sorted({year for hist in rank_history.values() for year in hist.keys()})

for year in all_years:
    # Build a list of (rank, capital) for this year
    year_data = []
    for rank in ranks_to_follow:
        if year in rank_history[rank]:
            cap = rank_history[rank][year]['capital']
            year_data.append((rank, cap))
    year_data.sort(key=lambda x: x[1], reverse=True)

    print(f"{year}:")
    for rank, cap in year_data:
        print(f"  Rank-{rank} | Capital: ${cap:.2f}")
    print()
