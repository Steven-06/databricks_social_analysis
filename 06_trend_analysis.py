import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})

SENT_COLORS = {
    "positive": "#378ADD",
    "neutral":  "#888780",
    "negative": "#D85A30",
}
CAT_COLORS = {
    "global_events":  "#1D9E75",
    "ai_and_tech":    "#378ADD",
    "entertainment":  "#D85A30",
    "economy":        "#BA7517",
    "sports":         "#888780",
}

# Load 
df = pd.read_csv("preprocessed_data.csv")
df["date"] = pd.to_datetime(df["date"])
print(f"Loaded {len(df):,} records | {df['date'].min().date()} → {df['date'].max().date()}")

# ============================================================
# Daily sentiment trends
# ============================================================
daily_sent = (
    df.groupby(["date", "sentiment"])
    .size()
    .unstack(fill_value=0)
    .sort_index()
)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Line chart
for sent in ["positive", "neutral", "negative"]:
    axes[0].plot(daily_sent.index, daily_sent[sent],
                 label=sent.capitalize(), color=SENT_COLORS[sent],
                 linewidth=1.8, marker="o", markersize=3)
axes[0].set_title("Daily post volume by sentiment", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Post count")
axes[0].legend(loc="upper right", fontsize=9)
axes[0].grid(axis="y", alpha=0.3)

# Stacked area chart
axes[1].stackplot(daily_sent.index,
                  daily_sent.get("positive", 0),
                  daily_sent.get("neutral", 0),
                  daily_sent.get("negative", 0),
                  labels=["Positive", "Neutral", "Negative"],
                  colors=[SENT_COLORS[s] for s in ["positive", "neutral", "negative"]],
                  alpha=0.75)
axes[1].set_title("Sentiment volume — stacked area", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Post count")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=4))
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right")
axes[1].legend(loc="upper right", fontsize=9)

plt.suptitle("Sentiment trends — February 2026", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("trend_01_sentiment_daily.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved trend_01_sentiment_daily.png")

# ============================================================
# Category activity over time
# ============================================================
daily_cat = (
    df.groupby(["date", "topic_category"])
    .size()
    .unstack(fill_value=0)
    .sort_index()
)

fig, ax = plt.subplots(figsize=(14, 5))
for cat in daily_cat.columns:
    ax.plot(daily_cat.index, daily_cat[cat],
            label=cat.replace("_", " ").title(),
            color=CAT_COLORS.get(cat, "#888780"),
            linewidth=1.8, marker="o", markersize=3)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax.set_title("Daily activity by topic category", fontsize=13, fontweight="bold")
ax.set_ylabel("Post count")
ax.legend(fontsize=9, ncol=2)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("trend_02_category_daily.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved trend_02_category_daily.png")

# ============================================================
# Engagement & trend score over time
# ============================================================
daily_eng   = df.groupby("date")["engagement_score"].mean()
daily_trend = df.groupby("date")["trend_score"].mean()

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

axes[0].bar(daily_eng.index, daily_eng.values,
            color="#378ADD", alpha=0.8, width=0.7)
axes[0].plot(daily_eng.index, daily_eng.rolling(7).mean(),
             color="#D85A30", linewidth=2, label="7-day MA")
axes[0].set_title("Avg daily engagement score", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Engagement score")
axes[0].legend(fontsize=9)
axes[0].grid(axis="y", alpha=0.3)

axes[1].bar(daily_trend.index, daily_trend.values,
            color="#1D9E75", alpha=0.8, width=0.7)
axes[1].plot(daily_trend.index, daily_trend.rolling(7).mean(),
             color="#D85A30", linewidth=2, label="7-day MA")
axes[1].set_title("Avg daily trend score", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Trend score")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=4))
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right")
axes[1].legend(fontsize=9)
axes[1].grid(axis="y", alpha=0.3)

plt.suptitle("Engagement & trend metrics over time", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("trend_03_engagement_time.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved trend_03_engagement_time.png")

# ============================================================
# Subcategory heat map over time
# ============================================================
# Aggregate by week × subcategory
df["week_label"] = df["date"].dt.strftime("W%V (%b %d)")
pivot = (
    df.groupby(["week_label", "topic_subcategory"])
    .size()
    .unstack(fill_value=0)
)

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(pivot.T, cmap="YlOrBr", ax=ax, linewidths=0.3,
            cbar_kws={"shrink": 0.6, "label": "Post count"})
ax.set_title("Subcategory activity by week", fontsize=13, fontweight="bold")
ax.set_xlabel("Week")
ax.set_ylabel("Subcategory")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("trend_04_subcat_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved trend_04_subcat_heatmap.png")

# ============================================================
# Top trending subcategories (bar race snapshot)
# ============================================================
# Engagement-weighted subcategory volume
subcat_eng = (
    df.groupby("topic_subcategory")
    .agg(count=("id", "count"), avg_engagement=("engagement_score", "mean"),
         avg_trend=("trend_score", "mean"))
    .sort_values("avg_engagement", ascending=False)
)

fig, axes = plt.subplots(1, 3, figsize=(15, 6))

axes[0].barh(subcat_eng.index[::-1], subcat_eng["count"][::-1],
             color="#378ADD", edgecolor="white", linewidth=0.5)
axes[0].set_title("Post volume", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Count")

axes[1].barh(subcat_eng.index[::-1], subcat_eng["avg_engagement"][::-1],
             color="#1D9E75", edgecolor="white", linewidth=0.5)
axes[1].set_title("Avg engagement score", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Score")
axes[1].set_yticklabels([])

axes[2].barh(subcat_eng.index[::-1], subcat_eng["avg_trend"][::-1],
             color="#D85A30", edgecolor="white", linewidth=0.5)
axes[2].set_title("Avg trend score", fontsize=11, fontweight="bold")
axes[2].set_xlabel("Score (0-1)")
axes[2].set_yticklabels([])

plt.suptitle("Subcategory performance metrics", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("trend_05_subcat_performance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved trend_05_subcat_performance.png")

# ============================================================
# Source comparison
# ============================================================
source_sent = pd.crosstab(df["source"], df["sentiment"])
source_eng  = df.groupby("source")["engagement_score"].mean()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

source_sent.plot(kind="bar", ax=axes[0],
                 color=[SENT_COLORS[s] for s in source_sent.columns],
                 edgecolor="white", linewidth=0.5)
axes[0].set_title("Sentiment by data source", fontsize=11, fontweight="bold")
axes[0].set_xlabel("")
axes[0].tick_params(axis="x", rotation=15)
axes[0].legend(title="Sentiment", fontsize=9)

source_eng.plot(kind="bar", ax=axes[1], color=["#1D9E75","#378ADD","#BA7517"],
                edgecolor="white", linewidth=0.5)
axes[1].set_title("Avg engagement by source", fontsize=11, fontweight="bold")
axes[1].set_xlabel("")
axes[1].tick_params(axis="x", rotation=15)
for i, v in enumerate(source_eng.values):
    axes[1].text(i, v + 30, f"{int(v):,}", ha="center", fontsize=9)

plt.suptitle("Data source analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("trend_06_source.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved trend_06_source.png")

# ============================================================
# Word clouds by topic category
# ============================================================
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
cats    = df["topic_category"].unique()
cmaps   = ["Greens", "Blues", "Oranges", "YlOrBr", "Greys"]

for i, (cat, cmap) in enumerate(zip(cats, cmaps)):
    text = " ".join(df[df["topic_category"] == cat]["processed_headline"].dropna())
    wc   = WordCloud(width=280, height=200, background_color="white",
                     colormap=cmap, max_words=40, collocations=False).generate(text)
    axes[i].imshow(wc, interpolation="bilinear")
    axes[i].axis("off")
    axes[i].set_title(cat.replace("_", " ").title(), fontsize=9, fontweight="bold")

plt.suptitle("Word clouds by topic category", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("trend_07_wordclouds_category.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved trend_07_wordclouds_category.png")

# ============================================================
# Correlation matrix
# ============================================================
numeric_cols = ["engagement_score", "trend_score", "lda_topic" if "lda_topic" in df.columns else "engagement_score"]
numeric_cols = [c for c in ["engagement_score", "trend_score"] if c in df.columns]
corr_df = df[numeric_cols].copy()
corr_df["sentiment_num"] = df["sentiment"].map({"negative": 0, "neutral": 1, "positive": 2})
corr_df["category_num"]  = df["topic_category"].astype("category").cat.codes

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax, linewidths=0.5, vmin=-1, vmax=1, center=0,
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature correlation matrix", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("trend_08_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved trend_08_correlation.png")

# ============================================================
# Text report
# ============================================================
peak_day = daily_sent.sum(axis=1).idxmax()
low_day  = daily_sent.sum(axis=1).idxmin()
top_sent = daily_sent.mean().idxmax()

with open("trend_report.txt", "w") as f:
    f.write("="*60 + "\n")
    f.write("TREND ANALYSIS REPORT — February 2026\n")
    f.write("="*60 + "\n\n")
    f.write(f"Total posts      : {len(df):,}\n")
    f.write(f"Date range       : {df['date'].min().date()} → {df['date'].max().date()}\n")
    f.write(f"Peak activity day: {peak_day.date()} ({int(daily_sent.sum(axis=1).max())} posts)\n")
    f.write(f"Lowest activity  : {low_day.date()} ({int(daily_sent.sum(axis=1).min())} posts)\n")
    f.write(f"Dominant sentiment (by avg daily volume): {top_sent}\n\n")
    f.write("─"*40 + "\n")
    f.write("Category volumes:\n")
    f.write(df["topic_category"].value_counts().to_string() + "\n\n")
    f.write("─"*40 + "\n")
    f.write("Subcategory performance (top 5 by engagement):\n")
    f.write(subcat_eng.head(5).to_string() + "\n\n")
    f.write("─"*40 + "\n")
    f.write("Avg engagement by sentiment:\n")
    f.write(df.groupby("sentiment")["engagement_score"].mean().round(1).to_string() + "\n\n")
    f.write("─"*40 + "\n")
    f.write("Avg trend score by sentiment:\n")
    f.write(df.groupby("sentiment")["trend_score"].mean().round(3).to_string() + "\n")

print("\nSaved trend_report.txt")
print("\nTrend analysis complete.")
