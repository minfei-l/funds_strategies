import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

mutual_fund_files = [
    "040001.xlsx", "050001.xlsx", "070002.xlsx", "110011.xlsx", "161005.xlsx",
    "163402.xlsx", "202002.xlsx", "270006.xlsx", "377010.xlsx", "260116.xlsx"
]
start_capital = 100
ranks_to_follow = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def calculate_max_drawdown(series):
    cumulative = (1 + series).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

# Load daily return
def get_daily_return(df):
    df = df[['净值日期', '日增长率']].copy()
    df.columns = ['date', 'daily_return']
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= '2012-01-01') & (df['date'] <= '2024-12-31')]
    df = df.sort_values('date')
    df['daily_return'] = pd.to_numeric(df['daily_return'], errors='coerce') / 100
    return df.dropna()

# Load all funds
fund_returns = {}
for filename in mutual_fund_files:
    fund_code = filename.split('.')[0]
    df = pd.read_excel(filename)
    fund_returns[fund_code] = get_daily_return(df)

# Create 6-month rebalancing windows
rebalance_dates = []
current_date = pd.to_datetime('2012-01-01')
while current_date + pd.DateOffset(months=6) <= pd.to_datetime('2025-01-01'):
    rebalance_dates.append(current_date)
    current_date += pd.DateOffset(months=6)

print([d.strftime('%Y-%m') for d in rebalance_dates])

# Simulate strategies
rank_capitals = {rank: start_capital for rank in ranks_to_follow}
rank_history = {rank: {} for rank in ranks_to_follow}
ranked_funds_history = {}

for i in range(1, len(rebalance_dates)):
    prev_date = rebalance_dates[i - 1]
    curr_date = rebalance_dates[i]
    hold_end = curr_date + pd.DateOffset(months=6)
    lookback_start = curr_date - pd.DateOffset(months=6)
    lookback_end = curr_date - pd.Timedelta(days=1)
    past_returns = {}

    for code, df in fund_returns.items():
        sub = df[(df['date'] >= lookback_start) & (df['date'] <= lookback_end)]
        if len(sub) < 15:
            continue
        past_returns[code] = (1 + sub['daily_return']).prod() - 1

    if len(past_returns) < max(ranks_to_follow):
        continue

    ranked_funds = sorted(past_returns.items(), key=lambda x: x[1], reverse=True)
    ranked_funds_history[curr_date] = ranked_funds

    for rank in ranks_to_follow:
        try:
            fund_code, _ = ranked_funds[rank - 1]
        except IndexError:
            continue

        df = fund_returns[fund_code]
        hold_data = df[(df['date'] >= curr_date) & (df['date'] < hold_end)]
        if len(hold_data) < 15:
            continue

        period_return = (1 + hold_data['daily_return']).prod() - 1
        capital = rank_capitals[rank] * (1 + period_return)
        rank_capitals[rank] = capital
        label = curr_date.strftime('%Y-%m')
        rank_history[rank][label] = {
            'fund': fund_code,
            'return': round(period_return * 100, 2),
            'capital': round(capital, 2)
        }

# Display Results
print("\n Strategy Results (Follow Top-N Fund Based on Past 6-Months):\n")
for rank in ranks_to_follow:
    print(f"️ Rank-{rank} Strategy:")
    for period in sorted(rank_history[rank].keys()):
        info = rank_history[rank][period]
        print(f"  {period}: Fund {info['fund']} | Return: {info['return']}% | Capital: ${info['capital']}")
    print(f"  ➤ Final Capital: ${rank_capitals[rank]:.2f}\n")

# lot Results
plt.figure(figsize=(12, 6))

for rank in ranks_to_follow:
    periods = sorted(rank_history[rank].keys())
    if not periods:
        continue

    # Prepend start capital at the first recorded period (simulate start point)
    start_period = pd.to_datetime(periods[0]) - pd.DateOffset(months=6)
    display_periods = [start_period.strftime('%Y-%m')] + periods
    capitals = [start_capital] + [rank_history[rank][p]['capital'] for p in periods if rank_history[rank][p]['capital'] is not None]

    # Plot the strategy with final capital in legend
    final_cap = capitals[-1]
    plt.plot(display_periods, capitals, marker='o', label=f'Rank-{rank} (${int(final_cap)})')

    # Annotate final capital
    # plt.text(display_periods[-1], capitals[-1], f"${capitals[-1]:.2f}", fontsize=9, verticalalignment='top')

plt.title("Capital Growth by Following Top-N Fund Every 6 Months\n(Based on Past 6-Month Return)")
plt.xlabel("Rebalance Period (YYYY-MM)")
plt.ylabel("Capital ($)")
plt.xticks(rotation=45)
plt.legend(title="Strategy (Final Capital)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Per-6-Month Volatility and Sharpe Ratio
print(" Per-6-Month Volatility and Sharpe Ratio (Rf = 0):\n")

results = []

for rank in ranks_to_follow:
    print(f"️ Rank-{rank} Strategy:")

    for period in sorted(rank_history[rank].keys()):
        fund_code = rank_history[rank][period]['fund']
        rebalance_date = pd.to_datetime(period)
        hold_end = rebalance_date + pd.DateOffset(months=6)

        df = fund_returns[fund_code]
        mask = (df['date'] >= rebalance_date) & (df['date'] < hold_end)
        hold_data = df.loc[mask]

        if hold_data.empty or len(hold_data) < 30:
            print(f"  {period}: Fund {fund_code} | Insufficient data")
            results.append({
                'Period': period,
                'Rank': rank,
                'Fund': fund_code,
                'Return (%)': None,
                'Volatility (%)': None,
                'Sharpe Ratio': None
            })
            continue

        daily_returns = hold_data['daily_return']
        volatility = daily_returns.std(ddof=1) * (252 ** 0.5)  # Annualized
        period_return = (1 + daily_returns).prod() - 1
        sharpe = period_return / volatility if volatility > 0 else float('nan')

        print(f"  {period}: Fund {fund_code} | Return: {period_return * 100:.2f}% | "
              f"Volatility: {volatility * 100:.2f}% | Sharpe: {sharpe:.2f}")

        results.append({
            'Period': period,
            'Rank': rank,
            'Fund': fund_code,
            'Return (%)': round(period_return * 100, 4),
            'Volatility (%)': round(volatility * 100, 4),
            'Sharpe Ratio': round(sharpe, 4)
        })

    print()

# Calculate Volatility, Sharpe, Max Drawdown
results = []
for rank in ranks_to_follow:
    for period in sorted(rank_history[rank].keys()):
        fund_code = rank_history[rank][period]['fund']
        rebalance_date = pd.to_datetime(period)
        hold_end = rebalance_date + pd.DateOffset(months=6)
        df = fund_returns[fund_code]
        hold_data = df[(df['date'] >= rebalance_date) & (df['date'] < hold_end)]

        if hold_data.empty or len(hold_data) < 30:
            results.append({
                'Period': period, 'Rank': rank, 'Fund': fund_code,
                'Return (%)': None, 'Volatility (%)': None,
                'Sharpe Ratio': None, 'Max Drawdown (%)': None
            })
            continue

        daily_returns = hold_data['daily_return']
        period_return = (1 + daily_returns).prod() - 1
        volatility = daily_returns.std(ddof=1) * (252 ** 0.5)
        sharpe = period_return / volatility if volatility > 0 else float('nan')
        max_dd = calculate_max_drawdown(daily_returns)

        results.append({
            'Period': period,
            'Rank': rank,
            'Fund': fund_code,
            'Return (%)': round(period_return * 100, 4),
            'Volatility (%)': round(volatility * 100, 4),
            'Sharpe Ratio': round(sharpe, 4),
            'Max Drawdown (%)': round(max_dd * 100, 4)
        })

# Save CSVs
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by=['Period', 'Rank'])
df_results[['Period', 'Rank', 'Fund', 'Return (%)']].to_csv("6month_growth.csv", index=False)
df_results[['Period', 'Rank', 'Fund', 'Volatility (%)']].to_csv("6month_volatility.csv", index=False)
df_results[['Period', 'Rank', 'Fund', 'Sharpe Ratio']].to_csv("6month_sharpe_ratio.csv", index=False)
df_results[['Period', 'Rank', 'Fund', 'Max Drawdown (%)']].to_csv("6month_max_drawdown.csv", index=False)

print("Saved to:")
print("  - '6month_growth.csv'")
print("  - '6month_volatility.csv'")
print("  - '6month_sharpe_ratio.csv'")
print("  - '6month_max_drawdown.csv'")

# Mean metrics by Rank
mean_metrics = df_results.groupby('Rank').agg({
    'Sharpe Ratio': 'mean',
    'Volatility (%)': 'mean',
    'Max Drawdown (%)': 'mean'
}).round(2)

# Create mappings for labeling
mean_sharpe = mean_metrics['Sharpe Ratio'].to_dict()
mean_vol = mean_metrics['Volatility (%)'].to_dict()
mean_mdd = mean_metrics['Max Drawdown (%)'].to_dict()

def plot_metric(df, metric_col, ylabel, title, filename, mean_map):
    plt.figure(figsize=(10, 6))
    for rank in sorted(df['Rank'].unique()):
        df_rank = df[df['Rank'] == rank]
        label = f'Rank-{rank} (Mean: {mean_map[rank]})'
        plt.plot(df_rank['Period'], df_rank[metric_col], marker='o', label=label)
    plt.title(title)
    plt.xlabel("Rebalance Period (YYYY-MM)")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    if "%" in ylabel:
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.grid(True)
    plt.legend(title="Strategy Rank")
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

plot_metric(df_results, 'Volatility (%)', "Volatility (%)", "6-Month Volatility by Strategy Rank", "volatility_plot.png", mean_vol)
plot_metric(df_results, 'Sharpe Ratio', "Sharpe Ratio", "6-Month Sharpe Ratio by Strategy Rank", "sharpe_plot.png", mean_sharpe)
plot_metric(df_results, 'Max Drawdown (%)', "Max Drawdown (%)", "6-Month Max Drawdown by Strategy Rank", "mdd_plot.png", mean_mdd)

# Load the 6-month CSVs
sharpe = pd.read_csv('6month_sharpe_ratio.csv')
volatility = pd.read_csv('6month_volatility.csv')
mdd = pd.read_csv('6month_max_drawdown.csv')

print("\n========== 6M SHARPE RATIO ==========")
print("Average Sharpe Ratio by Rank:")
print(sharpe.groupby('Rank')['Sharpe Ratio'].mean().sort_values(ascending=False))
best_sharpe = sharpe.loc[sharpe['Sharpe Ratio'].idxmax()]
print(f"\nBest Sharpe Ratio:\n  Period: {best_sharpe['Period']}, Rank: {best_sharpe['Rank']}, Value: {best_sharpe['Sharpe Ratio']:.3f}")
worst_sharpe = sharpe.loc[sharpe['Sharpe Ratio'].idxmin()]
print(f"Worst Sharpe Ratio:\n  Period: {worst_sharpe['Period']}, Rank: {worst_sharpe['Rank']}, Value: {worst_sharpe['Sharpe Ratio']:.3f}")

print("\n========== 6M VOLATILITY (%) ==========")
print("Average Volatility by Rank (%):")
print(volatility.groupby('Rank')['Volatility (%)'].mean().sort_values())
most_volatile = volatility.loc[volatility['Volatility (%)'].idxmax()]
print(f"\nMost Volatile:\n  Period: {most_volatile['Period']}, Rank: {most_volatile['Rank']}, Value: {most_volatile['Volatility (%)']:.2f}%")
least_volatile = volatility.loc[volatility['Volatility (%)'].idxmin()]
print(f"Least Volatile:\n  Period: {least_volatile['Period']}, Rank: {least_volatile['Rank']}, Value: {least_volatile['Volatility (%)']:.2f}%")

print("\n========== 6M MAX DRAWDOWN (%) ==========")
print("Average Max Drawdown by Rank (%):")
print(mdd.groupby('Rank')['Max Drawdown (%)'].mean().sort_values())
worst_drawdown = mdd.loc[mdd['Max Drawdown (%)'].idxmin()]
print(f"\nWorst Max Drawdown:\n  Period: {worst_drawdown['Period']}, Rank: {worst_drawdown['Rank']}, Value: {worst_drawdown['Max Drawdown (%)']:.2f}%")
best_drawdown = mdd.loc[mdd['Max Drawdown (%)'].idxmax()]
print(f"Smallest Max Drawdown:\n  Period: {best_drawdown['Period']}, Rank: {best_drawdown['Rank']}, Value: {best_drawdown['Max Drawdown (%)']:.2f}%")

# Load 6-month growth data
try:
    growth = pd.read_csv('6month_growth.csv')
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

# Show summary table
print("\n6-Month Summary stability/risk table:\n")
print(summary.sort_values(by=['Vol_Mean','MDD_Mean','Sharpe_Mean']))

print("\nBest by each metric:")
print("Lowest Volatility:", summary['Vol_Mean'].idxmin())
print("Lowest MDD:", summary['MDD_Mean'].idxmin())
print("Highest Sharpe:", summary['Sharpe_Mean'].idxmax())
if growth_available:
    print("Highest Mean Return:", summary['Return_Mean'].idxmax())
    print("Lowest Std Return:", summary['Return_Std'].idxmin())

