# Evaluating Simple Strategies with Mutual Funds and ETFs to Outperform the China’s Shanghai Composite Index (SCI)

This repository contains Python implementations of the allocation strategy, Shanghai Composite Index buy-and-hold (SCI B&H), Mutual Funds buy-and-hold (MF B&H) and top-n ranking strategy.

## Strategy Design and Evaluation

Five strategies were considered with the goal of identifying an approach that outperforms the Shanghai Composite Index:

1. Shanghai Composite Index Buy-and-Hold (SCI B&H)
   Used as the benchmark strategy.

2. Mutual Fund Buy-and-Hold (MF B&H)
   Buy-and-hold strategy applied to the ten largest (by assets) large-cap mutual funds over the period 2012–2024.

3. Allocation Strategies 
   Portfolios constructed by replicating each mutual fund’s prior-year sector allocations using sector ETFs.

4. Top-N Strategies
   Strategies that switch holdings among mutual funds based on their performance rankings in the previous year.

5. Simple Sector Rotation Strategies
   Sector-based rotation approaches using historical sector performance.

For each strategy, we evaluated:
- Capital growth
- Volatility
- Sharpe ratio
- Maximum drawdown (MDD)

---

## Repository Structure

```text
.
├── allocation-strategy-with-two-B&H/
│   ├── 3_strategies2.py
│   ├── all_capital_values.csv
│   ├── annual_mdd_by_fund.csv
│   ├── annual_sharpe_ratio_by_fund.csv
│   ├── annual_volatility_by_fund.csv
│   └── avg_metrics_by_code.csv
│
└── n-th-rank-strategy/
    ├── annual/
    │   ├── best_funds_annual.py
    │   ├── best_funds_annual3.py
    │   ├── annual_growth.csv
    │   ├── annual_returns_10_funds_T.csv
    │   ├── annual_returns_11_including_shcomp_T.csv
    │   ├── annual_rank_10_funds_T.csv
    │   ├── annual_rank_11_including_shcomp_T.csv
    │   ├── annual_volatility.csv
    │   ├── annual_sharpe_ratio.csv
    │   ├── annual_max_drawdown.csv
    │   ├── rolling3y_total_returns_10_funds_T.csv
    │   ├── rolling3y_total_returns_11_including_shcomp_T.csv
    │   ├── rolling3y_total_rank_10_funds_T.csv
    │   └── rolling3y_total_rank_11_including_shcomp_T.csv
    │
    ├── semi-annual/
    │   ├── best_funds_6m.py
    │   ├── 6month_growth.csv
    │   ├── 6month_volatility.csv
    │   ├── 6month_sharpe_ratio.csv
    │   └── 6month_max_drawdown.csv
    │
    └── quarter/
        ├── best_funds_3m.py
        ├── 3month_growth.csv
        ├── 3month_volatility.csv
        ├── 3month_sharpe_ratio.csv
        └── 3month_mdd.csv

