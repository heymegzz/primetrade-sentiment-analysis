# Primetrade.ai Internship Assignment
## Trader Behavior & Bitcoin Market Sentiment Analysis

> **Hyperliquid Historical Trades × Bitcoin Fear & Greed Index**  
> Submitted by: Meghna Nair| Data Science Internship Assignment

---

## Overview

This project investigates the relationship between Bitcoin market sentiment (Fear & Greed Index) and trader performance on Hyperliquid, a decentralized perpetuals exchange. The analysis goes beyond simple groupby summaries to deliver statistically validated insights, behavioral finance findings, trader segmentation, and a machine learning model — structured the way a junior quantitative researcher would approach it.

**Core question:** Does market sentiment meaningfully influence how traders behave, and does that behavior translate into measurable differences in performance?

**Short answer from the data:** Yes — with p = 1.2e-156 (Kruskal-Wallis), sentiment regimes produce statistically significant differences in PnL. But the direction is counterintuitive.

---

## Key Findings

| Finding | Detail |
|---|---|
| **Extreme Greed = Best Mean PnL** | $73.9 vs $57.1 in Extreme Fear — but position sizes are *smaller*, suggesting smart money gets cautious at the top |
| **Fear regime is highest volume** | 29,808 trades — traders are most active when the market feels worst |
| **Win rates are sentiment-driven** | Range from 76.2% (Extreme Fear) to 89.2% (Extreme Greed) |
| **Long bias flips with sentiment** | ~60% long in Fear, drops to ~33% in Extreme Greed — traders go contrarian at extremes |
| **FG Index has no lag predictive power** | Correlations at 1, 3, 7-day lags are near zero (r < 0.04) — the index alone isn't an alpha signal |
| **Random Forest AUC = 0.97** | Trade size + hour of day + coin are the dominant predictors of win/loss, not sentiment alone |
| **Cluster 3 traders are elite** | Mean PnL $279, Sharpe 0.75, win rate 87% — a small group driving outsized returns |
| **Loss aversion lowest in Extreme Greed** | Loss-to-gain ratio of 0.036 vs 0.185 in Extreme Fear — traders cut losses tighter when confident |

---

## Project Structure

```
primetrade-sentiment-analysis/
│
├── data/
│   ├── fear_greed_index.csv          # Bitcoin Fear & Greed Index (2018–2025)
│   └── historical_data.csv           # Hyperliquid trader history
│
├── notebooks/
│   └── analysis.py                   # Full end-to-end analysis pipeline
│
├── outputs/
│   ├── 01_sentiment_overview.png     # Trade volume + FG regime distribution
│   ├── 02_fg_timeline.png            # BTC Fear & Greed history (2018–2025)
│   ├── 03_mean_pnl_by_sentiment.png  # Mean closed PnL by sentiment regime
│   ├── 04_win_rate.png               # Win rate by sentiment
│   ├── 05_total_pnl.png              # Total cumulative PnL by sentiment
│   ├── 06_position_size_by_sentiment.png  # Trade size boxplot (overconfidence signal)
│   ├── 07_long_short_bias.png        # Long bias % by sentiment
│   ├── 08_pnl_violin.png             # PnL distributions + Kruskal-Wallis test
│   ├── 09_sharpe_by_sentiment.png    # Risk-adjusted returns by regime
│   ├── 10_loss_aversion.png          # Loss-to-gain ratio (behavioral finance)
│   ├── 11_coin_sentiment_heatmap.png # Mean PnL: top 10 coins × sentiment
│   ├── 12_intraday_pnl.png           # Hourly PnL patterns across regimes
│   ├── 13_monthly_pnl_fg.png         # Monthly PnL vs FG index overlay
│   ├── 14_elbow.png                  # K-Means elbow for trader segmentation
│   ├── 15_cluster_pnl_dist.png       # PnL distribution by trader cluster
│   ├── 16_cluster_sentiment_heatmap.png  # Cluster × sentiment interaction
│   ├── 17_fg_lag_analysis.png        # FG index lag predictive power (0–7 days)
│   ├── 18_feature_importance.png     # Random Forest feature importances
│   ├── 19_correlation_matrix.png     # Feature correlation heatmap
│   ├── core_metrics.csv              # PnL, win rate, Sharpe by sentiment
│   ├── risk_metrics.csv              # Sharpe + Calmar proxies by sentiment
│   ├── loss_aversion.csv             # Loss-to-gain ratios by sentiment
│   ├── pairwise_tests.csv            # Mann-Whitney U pairwise significance tests
│   └── trader_clusters.csv           # K-Means cluster profiles
│
└── README.md
```

---

## Methodology

### Data Sources
- **Fear & Greed Index:** Daily sentiment classification (Extreme Fear / Fear / Neutral / Greed / Extreme Greed) with numeric value 0–100
- **Hyperliquid Trades:** Individual trade records including account, coin, execution price, size (tokens + USD), direction, closed PnL, fee, and IST timestamp

### Data Cleaning
- Parsed IST timestamps (`DD-MM-YYYY HH:MM` format)
- Winsorised PnL at 1st/99th percentile to remove outlier distortion
- Filtered to closed trades only (PnL ≠ 0) for performance analysis
- Merged on date key; dropped ~0 unmatched rows (near-complete overlap)

### Feature Engineering
- `is_win` — binary win flag (PnL > 0)
- `is_long` — direction parsed from trade type string
- `is_liquidation` — flagged from direction field
- `pnl_per_usd` — return per dollar deployed
- `hour` — extracted from IST timestamp for intraday analysis
- `fg_lag1/3/7` — lagged FG values for predictive testing
- Per-account aggregations: total PnL, win rate, Sharpe proxy, liquidation rate, long bias

### Statistical Tests Used
- **Kruskal-Wallis H-test** — non-parametric ANOVA for PnL differences across 5 sentiment regimes (correct choice given non-normal PnL distributions)
- **Mann-Whitney U** — pairwise significance testing between all regime pairs
- **Pearson correlation** — FG index lag analysis at 0, 1, 3, 7-day horizons

### Machine Learning
- **Random Forest Classifier** — predicts trade win/loss; 5-fold stratified CV; AUC = 0.97
- **Gradient Boosting Classifier** — cross-validated comparison model
- **Logistic Regression** — baseline linear model
- Features: trade size (USD), hour of day, FG value, coin, sentiment code, direction

### Trader Segmentation
- K-Means clustering on per-account features (mean PnL, win rate, PnL volatility, liquidation rate, long bias, Sharpe, mean size)
- Elbow method selected K=4
- Clusters interpreted as: Cautious Low-Volume, High-Frequency Directional, Balanced Mid-Tier, Elite High-Sharpe

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/primetrade-sentiment-analysis.git
cd primetrade-sentiment-analysis
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels
```

### 3. Add data files
Place both CSVs in the `data/` folder:
- `data/fear_greed_index.csv`
- `data/historical_data.csv`

### 4. Run the analysis
```bash
cd notebooks
python analysis.py
```

All 19 charts and 5 CSV tables will be saved to `outputs/`.

---

## Results Summary

### Core Metrics by Sentiment Regime

| Sentiment | Trades | Mean PnL | Win Rate | Sharpe Proxy |
|---|---|---|---|---|
| Extreme Fear | 10,406 | $57.1 | 76.2% | 0.307 |
| Fear | 29,808 | $66.5 | 87.3% | 0.347 |
| Neutral | 18,159 | $54.9 | 82.4% | 0.312 |
| Greed | 25,176 | $59.3 | 76.9% | 0.329 |
| Extreme Greed | 20,853 | $73.9 | 89.2% | 0.381 |

### Trader Cluster Profiles

| Cluster | Mean PnL | Win Rate | Sharpe | Mean Size USD | Profile |
|---|---|---|---|---|---|
| 0 | $89.7 | 75.2% | 0.39 | $2,295 | Low-size, moderate performance |
| 1 | $133.9 | 93.4% | 0.55 | $8,767 | High win-rate, large positions |
| 2 | $40.2 | 78.2% | 0.27 | $5,051 | Average, high volume |
| 3 | $279.2 | 87.1% | 0.75 | $5,942 | Elite — highest Sharpe and PnL |

### Statistical Significance
- **Kruskal-Wallis:** H = 728.9, p = 1.2e-156 — sentiment regimes produce highly significant PnL differences
- **Significant pairwise pairs:** Extreme Fear vs Fear, Fear vs Neutral, Fear vs Greed, Fear vs Extreme Greed, Neutral vs Extreme Greed, Greed vs Extreme Greed
- **Non-significant pairs:** Extreme Fear vs Greed (p=0.06), Neutral vs Greed (p=0.57) — honest reporting, not all regimes are separable

---

## Behavioral Finance Insights

**Overconfidence during Greed:** Position sizes are largest during Fear ($8,041 mean) and smallest during Extreme Greed ($2,780) — experienced traders actually reduce exposure at sentiment peaks, the opposite of what behavioral theory predicts for retail traders.

**Loss aversion is sentiment-dependent:** The loss-to-gain ratio is highest in Extreme Fear (0.185) and lowest in Extreme Greed (0.036) — traders hold losers longer when fearful and cut them faster when confident.

**Contrarian long/short behavior:** Traders are most long (~60%) during Extreme Fear and least long (~33%) during Extreme Greed — a contrarian positioning pattern that partially explains the superior mean PnL in fear regimes.

**FG Index has no short-term alpha:** Lag correlations at 1, 3, and 7 days are all below r = 0.04 with p > 0.4. The index reflects current sentiment but does not predict next-day trade outcomes.

---

## Dependencies

```
pandas >= 1.5
numpy >= 1.23
matplotlib >= 3.6
seaborn >= 0.12
scikit-learn >= 1.1
scipy >= 1.9
statsmodels >= 0.13
```

---

## Assignment Context

Submitted as part of the Primetrade.ai Data Science Internship hiring process. The objective was to explore the relationship between Bitcoin market sentiment and trader performance on Hyperliquid, uncover hidden patterns, and deliver insights that can drive smarter trading strategies.
