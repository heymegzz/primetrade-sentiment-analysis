"""
analysis.py
Primetrade.ai Internship Assignment
Trader Behavior & Market Sentiment Analysis
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import os

# ── PATHS ── change these if your folder structure differs ──────────────────
# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

FG_PATH     = os.path.join(PROJECT_ROOT, "data", "fear_greed_index.csv")
TRADES_PATH = os.path.join(PROJECT_ROOT, "data", "historical_data.csv")
OUT_DIR     = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── PLOT STYLE ───────────────────────────────────────────────────────────────
SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
PALETTE = {
    "Extreme Fear":  "#c0392b",
    "Fear":          "#e67e22",
    "Neutral":       "#95a5a6",
    "Greed":         "#27ae60",
    "Extreme Greed": "#1a5276"
}
sns.set_theme(style="darkgrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight",
                     "savefig.facecolor": "white"})

def save(name):
    plt.savefig(f"{OUT_DIR}/{name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {name}.png")

# ════════════════════════════════════════════════════════════════════════════
# 1. LOAD & CLEAN
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 1. LOADING DATA ===")

fg = pd.read_csv(FG_PATH)
fg["date"] = pd.to_datetime(fg["date"]).dt.date  # keep as date only

trades = pd.read_csv(TRADES_PATH)

# parse the IST timestamp — format is DD-MM-YYYY HH:MM
trades["datetime"] = pd.to_datetime(trades["Timestamp IST"], format="%d-%m-%Y %H:%M", errors="coerce")
trades["date"]     = trades["datetime"].dt.date
trades["hour"]     = trades["datetime"].dt.hour

# rename for convenience
trades = trades.rename(columns={
    "Account":         "account",
    "Coin":            "coin",
    "Execution Price": "exec_price",
    "Size Tokens":     "size_tokens",
    "Size USD":        "size_usd",
    "Side":            "side",
    "Start Position":  "start_pos",
    "Direction":       "direction",
    "Closed PnL":      "pnl",
    "Fee":             "fee",
})

# drop rows where date parsing failed
trades = trades.dropna(subset=["datetime"])

# winsorise PnL at 1/99 percentile
p1, p99 = trades["pnl"].quantile([0.01, 0.99])
trades["pnl_raw"] = trades["pnl"].copy()
trades["pnl"]     = trades["pnl"].clip(p1, p99)

# filter to only actual closed trades with real PnL for most analyses
# (many rows are partial fills with pnl=0; keep all for volume analysis)
closed = trades[trades["pnl"] != 0].copy()

print(f"  Total rows         : {len(trades):,}")
print(f"  Closed PnL rows    : {len(closed):,}")
print(f"  Unique accounts    : {trades['account'].nunique()}")
print(f"  Unique coins       : {trades['coin'].nunique()}")
print(f"  Date range         : {trades['date'].min()} -> {trades['date'].max()}")

# ════════════════════════════════════════════════════════════════════════════
# 2. MERGE WITH FEAR & GREED
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 2. MERGING ===")

fg_lookup = fg.set_index("date")[["value", "classification"]].to_dict("index")

trades["fg_value"]  = trades["date"].map(lambda d: fg_lookup.get(d, {}).get("value"))
trades["sentiment"] = trades["date"].map(lambda d: fg_lookup.get(d, {}).get("classification"))

closed["fg_value"]  = closed["date"].map(lambda d: fg_lookup.get(d, {}).get("value"))
closed["sentiment"] = closed["date"].map(lambda d: fg_lookup.get(d, {}).get("classification"))

# drop rows with no sentiment match
before = len(closed)
closed = closed.dropna(subset=["sentiment"])
print(f"  Closed rows after FG merge: {len(closed):,} (dropped {before - len(closed)} unmatched)")

closed["sentiment"] = pd.Categorical(closed["sentiment"], categories=SENTIMENT_ORDER, ordered=True)

# derived columns
closed["is_win"]          = (closed["pnl"] > 0).astype(int)
closed["is_liquidation"]  = closed["direction"].str.contains("Liquidat", case=False, na=False)
closed["is_long"]         = closed["direction"].str.contains("Long|Buy", case=False, na=False)
closed["pnl_per_usd"]     = closed["pnl"] / closed["size_usd"].replace(0, np.nan)

print("  Sentiment distribution:")
print(closed["sentiment"].value_counts().reindex(SENTIMENT_ORDER).to_string())

# ════════════════════════════════════════════════════════════════════════════
# 3. SENTIMENT OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 3. EDA CHARTS ===")

# Chart 1: trade count by sentiment
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
counts = closed["sentiment"].value_counts().reindex(SENTIMENT_ORDER)
axes[0].bar(counts.index, counts.values, color=[PALETTE[s] for s in counts.index])
axes[0].set_title("Trade Volume by Sentiment Regime", fontweight="bold")
axes[0].set_ylabel("Number of Trades")
axes[0].tick_params(axis="x", rotation=20)

fg_day_counts = fg["classification"].value_counts().reindex(SENTIMENT_ORDER)
axes[1].pie(fg_day_counts.values, labels=fg_day_counts.index,
            colors=[PALETTE[s] for s in fg_day_counts.index],
            autopct="%1.0f%%", startangle=140)
axes[1].set_title("Days in Each Sentiment Regime (Full FG History)", fontweight="bold")
plt.tight_layout()
save("01_sentiment_overview")

# Chart 2: FG index timeline
fg_plot = fg.copy()
fg_plot["date"] = pd.to_datetime(fg_plot["date"])
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(fg_plot["date"], fg_plot["value"], lw=1, color="#2c3e50", alpha=0.8)
ax.axhspan(0,  25, alpha=0.12, color="#c0392b", label="Extreme Fear")
ax.axhspan(25, 45, alpha=0.10, color="#e67e22", label="Fear")
ax.axhspan(45, 55, alpha=0.08, color="#95a5a6", label="Neutral")
ax.axhspan(55, 75, alpha=0.10, color="#27ae60", label="Greed")
ax.axhspan(75,100, alpha=0.12, color="#1a5276", label="Extreme Greed")
ax.set_title("Bitcoin Fear & Greed Index — Full History (2018–2025)", fontweight="bold")
ax.set_ylabel("FG Value"); ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
save("02_fg_timeline")

# ════════════════════════════════════════════════════════════════════════════
# 4. CORE PERFORMANCE METRICS BY SENTIMENT
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 4. CORE METRICS ===")

core = closed.groupby("sentiment", observed=True).agg(
    trade_count    = ("pnl", "count"),
    mean_pnl       = ("pnl", "mean"),
    median_pnl     = ("pnl", "median"),
    total_pnl      = ("pnl", "sum"),
    win_rate       = ("is_win", "mean"),
    pnl_std        = ("pnl", "std"),
    mean_size_usd  = ("size_usd", "mean"),
    liq_rate       = ("is_liquidation", "mean"),
    mean_fee       = ("fee", "mean"),
).reset_index()

core["sharpe_proxy"] = core["mean_pnl"] / core["pnl_std"]
core.to_csv(f"{OUT_DIR}/core_metrics.csv", index=False)
print(core[["sentiment","trade_count","mean_pnl","win_rate","liq_rate","sharpe_proxy"]].to_string(index=False))

# Chart 3: Mean PnL by sentiment
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(core["sentiment"].astype(str), core["mean_pnl"],
              color=[PALETTE[s] for s in core["sentiment"]])
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_title("Mean Closed PnL by Sentiment Regime", fontweight="bold")
ax.set_ylabel("Mean PnL (USD)")
for bar, val in zip(bars, core["mean_pnl"]):
    ypos = bar.get_height() + (2 if val >= 0 else -8)
    ax.text(bar.get_x() + bar.get_width()/2, ypos, f"${val:.1f}",
            ha="center", fontsize=9, fontweight="bold")
plt.tight_layout()
save("03_mean_pnl_by_sentiment")

# Chart 4: Win rate
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(core["sentiment"].astype(str), core["win_rate"]*100,
       color=[PALETTE[s] for s in core["sentiment"]])
ax.axhline(50, color="black", lw=1, ls="--", label="50% baseline")
ax.set_title("Win Rate by Sentiment Regime", fontweight="bold")
ax.set_ylabel("Win Rate (%)")
ax.legend(); ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
save("04_win_rate")

# Chart 5: Total PnL by sentiment (who makes the most money overall)
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(core["sentiment"].astype(str), core["total_pnl"],
       color=[PALETTE[s] for s in core["sentiment"]])
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_title("Total Cumulative PnL by Sentiment Regime", fontweight="bold")
ax.set_ylabel("Total PnL (USD)")
ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
save("05_total_pnl")

# ════════════════════════════════════════════════════════════════════════════
# 5. POSITION SIZE ANALYSIS (proxy for leverage/risk appetite)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 5. POSITION SIZE / RISK APPETITE ===")

# Chart 6: Position size boxplot by sentiment
fig, ax = plt.subplots(figsize=(12, 5))
bp_data = [closed.loc[closed["sentiment"]==s, "size_usd"].dropna().values
           for s in SENTIMENT_ORDER]
bp = ax.boxplot(bp_data, patch_artist=True,
                notch=False, showfliers=False,
                medianprops=dict(color="white", lw=2))
ax.set_xticklabels(SENTIMENT_ORDER)
for patch, s in zip(bp["boxes"], SENTIMENT_ORDER):
    patch.set_facecolor(PALETTE[s]); patch.set_alpha(0.8)
ax.set_title("Trade Size (USD) by Sentiment — Overconfidence Signal", fontweight="bold")
ax.set_ylabel("Trade Size USD")
ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
save("06_position_size_by_sentiment")

# mean size table
size_by_sent = closed.groupby("sentiment", observed=True)["size_usd"].agg(["mean","median"])
print(size_by_sent.reindex(SENTIMENT_ORDER).round(2).to_string())

# ════════════════════════════════════════════════════════════════════════════
# 6. DIRECTION BIAS (Long vs Short across sentiment)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 6. DIRECTION BIAS ===")

dir_bias = closed.groupby("sentiment", observed=True)["is_long"].mean().reindex(SENTIMENT_ORDER)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(dir_bias.index.astype(str), dir_bias.values*100,
       color=[PALETTE[s] for s in dir_bias.index])
ax.axhline(50, color="black", lw=1, ls="--", label="50% neutral")
ax.set_title("Long Bias (%) by Sentiment — Do Traders Chase the Market?", fontweight="bold")
ax.set_ylabel("% Long Trades")
ax.legend(); ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
save("07_long_short_bias")

# ════════════════════════════════════════════════════════════════════════════
# 7. PnL DISTRIBUTION — VIOLIN PLOTS + STATISTICAL TEST
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 7. STATISTICAL TESTING ===")

groups = [closed.loc[closed["sentiment"]==s, "pnl"].values for s in SENTIMENT_ORDER]
h_stat, p_kw = kruskal(*[g for g in groups if len(g) > 0])
print(f"  Kruskal-Wallis: H={h_stat:.2f}, p={p_kw:.4e}")

fig, ax = plt.subplots(figsize=(13, 6))
valid_groups = [g for g in groups if len(g) > 1]
valid_labels = [s for s, g in zip(SENTIMENT_ORDER, groups) if len(g) > 1]
parts = ax.violinplot(valid_groups, positions=range(len(valid_labels)),
                      showmedians=True, showextrema=True)
for pc, s in zip(parts["bodies"], valid_labels):
    pc.set_facecolor(PALETTE[s]); pc.set_alpha(0.7)
ax.set_xticks(range(len(valid_labels)))
ax.set_xticklabels(valid_labels, rotation=15)
ax.set_title("PnL Distribution by Sentiment Regime", fontweight="bold")
ax.set_ylabel("Closed PnL (USD, winsorised)")
ax.axhline(0, color="black", lw=0.8, ls="--")
note = f"Kruskal-Wallis H={h_stat:.1f}, p={p_kw:.3e}"
ax.text(0.01, 0.97, note, transform=ax.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
plt.tight_layout()
save("08_pnl_violin")

# Pairwise Mann-Whitney
print("  Pairwise tests (significant pairs):")
pairs = []
for i, s1 in enumerate(SENTIMENT_ORDER):
    for j, s2 in enumerate(SENTIMENT_ORDER):
        if j <= i: continue
        g1 = closed.loc[closed["sentiment"]==s1, "pnl"].values
        g2 = closed.loc[closed["sentiment"]==s2, "pnl"].values
        if len(g1) < 5 or len(g2) < 5: continue
        u, p = mannwhitneyu(g1, g2, alternative="two-sided")
        pairs.append({"pair": f"{s1} vs {s2}", "p": round(p,4), "sig": p < 0.05})
pairwise_df = pd.DataFrame(pairs)
pairwise_df.to_csv(f"{OUT_DIR}/pairwise_tests.csv", index=False)
print(pairwise_df[pairwise_df["sig"]].to_string(index=False))

# ════════════════════════════════════════════════════════════════════════════
# 8. RISK-ADJUSTED METRICS
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 8. RISK-ADJUSTED ===")

risk = closed.groupby("sentiment", observed=True).agg(
    mean_pnl  = ("pnl","mean"),
    std_pnl   = ("pnl","std"),
    worst_pnl = ("pnl","min"),
).reset_index()
risk["sharpe"] = risk["mean_pnl"] / risk["std_pnl"]
risk["calmar"] = risk["mean_pnl"] / (-risk["worst_pnl"] + 1e-9)
risk.to_csv(f"{OUT_DIR}/risk_metrics.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(risk["sentiment"].astype(str), risk["sharpe"],
       color=[PALETTE[s] for s in risk["sentiment"]])
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_title("Sharpe Proxy (Mean PnL / Std PnL) by Sentiment", fontweight="bold")
ax.set_ylabel("Sharpe Proxy")
ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
save("09_sharpe_by_sentiment")

# ════════════════════════════════════════════════════════════════════════════
# 9. LOSS AVERSION
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 9. BEHAVIORAL FINANCE ===")

closed["abs_loss"] = closed["pnl"].apply(lambda x: abs(x) if x < 0 else 0)
closed["abs_gain"] = closed["pnl"].apply(lambda x: x if x > 0 else 0)

la = closed.groupby("sentiment", observed=True).agg(
    mean_loss=("abs_loss","mean"),
    mean_gain=("abs_gain","mean"),
).reset_index()
la["loss_gain_ratio"] = la["mean_loss"] / la["mean_gain"].replace(0, np.nan)
la.to_csv(f"{OUT_DIR}/loss_aversion.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(la["sentiment"].astype(str), la["loss_gain_ratio"],
       color=[PALETTE[s] for s in la["sentiment"]])
ax.axhline(1.0, color="black", lw=1, ls="--", label="Break-even ratio")
ax.set_title("Loss-to-Gain Ratio by Sentiment (Loss Aversion Proxy)", fontweight="bold")
ax.set_ylabel("|Mean Loss| / Mean Gain")
ax.legend(); ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
save("10_loss_aversion")

# ════════════════════════════════════════════════════════════════════════════
# 10. TOP COINS ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 10. COIN ANALYSIS ===")

# top 10 coins by trade count
top_coins = closed["coin"].value_counts().head(10).index.tolist()
coin_sent = (closed[closed["coin"].isin(top_coins)]
             .groupby(["coin","sentiment"], observed=True)["pnl"].mean()
             .unstack().reindex(columns=SENTIMENT_ORDER))

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(coin_sent, annot=True, fmt=".0f", cmap="RdYlGn",
            center=0, linewidths=0.5, ax=ax)
ax.set_title("Mean PnL by Coin × Sentiment (Top 10 Coins)", fontweight="bold")
plt.tight_layout()
save("11_coin_sentiment_heatmap")

# ════════════════════════════════════════════════════════════════════════════
# 11. TEMPORAL PATTERNS
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 11. TEMPORAL PATTERNS ===")

# Intraday PnL by sentiment
hourly = (closed.groupby(["hour","sentiment"], observed=True)["pnl"]
          .mean().unstack().reindex(columns=SENTIMENT_ORDER))

fig, ax = plt.subplots(figsize=(14, 5))
for col in SENTIMENT_ORDER:
    if col in hourly.columns:
        ax.plot(hourly.index, hourly[col], label=col,
                color=PALETTE[col], lw=2, marker="o", markersize=3)
ax.set_title("Average PnL by Hour of Day (IST) Across Sentiment Regimes", fontweight="bold")
ax.set_xlabel("Hour (IST)"); ax.set_ylabel("Mean PnL (USD)")
ax.legend(fontsize=9); ax.axhline(0, color="gray", lw=0.8, ls="--")
plt.tight_layout()
save("12_intraday_pnl")

# Monthly PnL + FG overlay
closed["month"] = closed["datetime"].dt.to_period("M")
monthly = closed.groupby("month")["pnl"].sum().reset_index()
monthly["month_dt"] = monthly["month"].dt.to_timestamp()

fg_plot2 = fg.copy()
fg_plot2["date"] = pd.to_datetime(fg_plot2["date"])
monthly_fg = fg_plot2.set_index("date").resample("ME")["value"].mean().reset_index()

fig, ax1 = plt.subplots(figsize=(16, 5))
ax2 = ax1.twinx()
ax1.bar(monthly["month_dt"], monthly["pnl"], width=20,
        color="#3498db", alpha=0.7, label="Monthly Total PnL")
ax2.plot(monthly_fg["date"], monthly_fg["value"],
         color="#e74c3c", lw=2, label="Avg FG Index")
ax1.set_ylabel("Total PnL (USD)", color="#3498db")
ax2.set_ylabel("Fear & Greed Index", color="#e74c3c")
ax1.set_title("Monthly Total PnL vs Fear & Greed Index", fontweight="bold")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc="upper left")
plt.tight_layout()
save("13_monthly_pnl_fg")

# ════════════════════════════════════════════════════════════════════════════
# 12. TRADER SEGMENTATION (per-account clustering)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 12. TRADER CLUSTERING ===")

trader = closed.groupby("account").agg(
    total_pnl     = ("pnl","sum"),
    mean_pnl      = ("pnl","mean"),
    win_rate      = ("is_win","mean"),
    trade_count   = ("pnl","count"),
    pnl_std       = ("pnl","std"),
    liq_rate      = ("is_liquidation","mean"),
    long_bias     = ("is_long","mean"),
    mean_size_usd = ("size_usd","mean"),
    mean_fee      = ("fee","mean"),
).reset_index()

trader["sharpe"] = trader["mean_pnl"] / trader["pnl_std"].replace(0, np.nan)
trader.fillna(0, inplace=True)

feats = ["mean_pnl","win_rate","pnl_std","liq_rate","long_bias","sharpe","mean_size_usd"]
X = StandardScaler().fit_transform(trader[feats].values)

# Elbow
inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_
            for k in range(2, 8)]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(2,8), inertias, "bo-", lw=2)
ax.axvline(4, color="red", ls="--", alpha=0.6, label="K=4")
ax.set_xlabel("K"); ax.set_ylabel("Inertia")
ax.set_title("K-Means Elbow — Trader Segmentation", fontweight="bold")
ax.legend(); plt.tight_layout()
save("14_elbow")

km = KMeans(n_clusters=4, random_state=42, n_init=10)
trader["cluster"] = km.fit_predict(X)

profiles = trader.groupby("cluster")[feats].mean().round(3)
profiles.to_csv(f"{OUT_DIR}/trader_clusters.csv")
print(profiles.to_string())

# cluster PnL distribution
fig, ax = plt.subplots(figsize=(10, 5))
for c in range(4):
    data = trader[trader["cluster"]==c]["total_pnl"]
    ax.hist(data, bins=20, alpha=0.6, label=f"Cluster {c}")
ax.set_title("Total PnL Distribution by Trader Cluster", fontweight="bold")
ax.set_xlabel("Total PnL (USD)"); ax.set_ylabel("# Traders")
ax.legend(); plt.tight_layout()
save("15_cluster_pnl_dist")

# merge cluster back and do cluster × sentiment heatmap
closed = closed.merge(trader[["account","cluster"]], on="account", how="left")
heat = (closed.groupby(["cluster","sentiment"], observed=True)["pnl"]
        .mean().unstack().reindex(columns=SENTIMENT_ORDER))

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(heat, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
            linewidths=0.5, ax=ax)
ax.set_title("Mean PnL: Trader Cluster × Sentiment Regime", fontweight="bold")
plt.tight_layout()
save("16_cluster_sentiment_heatmap")

# ════════════════════════════════════════════════════════════════════════════
# 13. FG LAG ANALYSIS — does yesterday's sentiment predict today's PnL?
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 13. FG LAG ANALYSIS ===")

daily_pnl = closed.groupby("date")["pnl"].mean().reset_index()
daily_pnl["date"] = pd.to_datetime(daily_pnl["date"])

fg_daily = fg.copy()
fg_daily["date"] = pd.to_datetime(fg_daily["date"])
fg_daily = fg_daily.sort_values("date")
fg_daily["fg_lag1"] = fg_daily["value"].shift(1)
fg_daily["fg_lag3"] = fg_daily["value"].shift(3)
fg_daily["fg_lag7"] = fg_daily["value"].shift(7)

lag_df = daily_pnl.merge(fg_daily[["date","value","fg_lag1","fg_lag3","fg_lag7"]],
                          on="date", how="inner").dropna()

print("  Lag correlations with daily mean PnL:")
for col in ["value","fg_lag1","fg_lag3","fg_lag7"]:
    r, p = stats.pearsonr(lag_df[col], lag_df["pnl"])
    print(f"    {col:10s}: r={r:.4f}, p={p:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, col, title in zip(axes.flat,
    ["value","fg_lag1","fg_lag3","fg_lag7"],
    ["Same Day","Lag 1 Day","Lag 3 Days","Lag 7 Days"]):
    ax.scatter(lag_df[col], lag_df["pnl"], alpha=0.35, s=15, color="#3498db")
    m, b, r, p, _ = stats.linregress(lag_df[col], lag_df["pnl"])
    xs = np.linspace(lag_df[col].min(), lag_df[col].max(), 100)
    ax.plot(xs, m*xs+b, color="#e74c3c", lw=2)
    ax.set_title(f"{title}: r={r:.3f}, p={p:.3f}", fontweight="bold")
    ax.set_xlabel("FG Index"); ax.set_ylabel("Daily Avg PnL")
plt.suptitle("FG Index Lag Analysis — Predictive Power", fontweight="bold", y=1.01)
plt.tight_layout()
save("17_fg_lag_analysis")

# ════════════════════════════════════════════════════════════════════════════
# 14. MACHINE LEARNING — predict win/loss
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 14. MACHINE LEARNING ===")

ml = closed.dropna(subset=["fg_value","size_usd"]).copy()
ml["coin_code"] = pd.factorize(ml["coin"])[0]
ml["sent_code"] = ml["sentiment"].cat.codes

feat_cols = ["fg_value","size_usd","is_long","coin_code","sent_code","hour"]
X_ml = ml[feat_cols].fillna(0).values
y_ml = ml["is_win"].values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "Logistic Regression":  LogisticRegression(max_iter=500, random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_ml, y_ml, cv=cv, scoring="roc_auc")
    results[name] = scores.mean()
    print(f"  {name}: AUC = {scores.mean():.4f} ± {scores.std():.4f}")

# Feature importance (RF)
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_ml, y_ml)
fi = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
fi.plot(kind="barh", ax=ax, color="#3498db")
ax.set_title("Feature Importance — Predicting Trade Win/Loss\n(Random Forest)", fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
save("18_feature_importance")

# ════════════════════════════════════════════════════════════════════════════
# 15. CORRELATION MATRIX
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 15. CORRELATION MATRIX ===")

corr_cols = ["pnl","size_usd","fg_value","is_win","is_long","fee","is_liquidation"]
corr_mat = closed[corr_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, mask=mask, ax=ax, linewidths=0.5)
ax.set_title("Feature Correlation Matrix", fontweight="bold")
plt.tight_layout()
save("19_correlation_matrix")

print("\n✅ ALL DONE. Check your outputs/ folder.")
print(f"   Charts saved: 19 PNG files")
print(f"   Tables saved: core_metrics.csv, risk_metrics.csv, trader_clusters.csv, etc.")