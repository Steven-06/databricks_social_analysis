import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import warnings
warnings.filterwarnings("ignore")

# Color Palette 
COLORS = {
    "positive": "#378ADD",
    "neutral":  "#888780",
    "negative": "#D85A30",
}
CAT_COLORS = ["#1D9E75", "#378ADD", "#D85A30", "#BA7517", "#888780"]
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})

# Load data 

# Local:
df = pd.read_csv("preprocessed_data.csv")
df["date"] = pd.to_datetime(df["date"])
print(f"Dataset shape: {df.shape}")
print(df.dtypes)

# Stopwords
STOPWORDS = {
    "i","me","my","we","our","you","your","he","him","she","her","it","its",
    "they","them","what","which","who","this","that","these","those","am","is",
    "are","was","were","be","been","being","have","has","had","do","does","did",
    "a","an","the","and","but","if","or","as","until","while","of","at","by",
    "for","with","about","into","through","during","before","after","to","from",
    "up","down","in","out","on","off","over","under","then","here","there",
    "when","where","why","how","all","both","each","few","more","most","other",
    "some","no","not","only","same","so","than","too","very","can","will",
    "just","now","users","actively","discussing","broader","category","across",
    "multiple","platforms","within","are","global","social","media","news",
    "also","latest","new","among",
}

def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(w for w in text.split() if w not in STOPWORDS and len(w) > 2)

# Use combined cleaned text if available, else we recompute
if "cleaned_headline" not in df.columns:
    df["cleaned_headline"] = df["headline"].apply(clean)
if "processed_headline" not in df.columns:
    df["processed_headline"] = df["cleaned_headline"]

df["combined_text"] = df["cleaned_headline"].fillna("") + " " + df.get("processed_headline", df["cleaned_headline"]).fillna("")
df["text_length"]   = df["headline"].str.len()
df["word_count"]    = df["headline"].str.split().str.len()


# Sentiment distribution

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Sentiment distribution", fontsize=14, fontweight="bold", y=1.02)

sent_counts = df["sentiment"].value_counts()

# Bar chart
axes[0].bar(sent_counts.index, sent_counts.values,
            color=[COLORS[s] for s in sent_counts.index], edgecolor="white", linewidth=0.5)
axes[0].set_title("Count per class")
axes[0].set_ylabel("Records")
for i, v in enumerate(sent_counts.values):
    axes[0].text(i, v + 5, str(v), ha="center", fontsize=10)

# Pie chart
axes[1].pie(sent_counts.values, labels=sent_counts.index,
            colors=[COLORS[s] for s in sent_counts.index],
            autopct="%1.1f%%", startangle=90, wedgeprops={"linewidth": 0.5, "edgecolor": "white"})
axes[1].set_title("Proportion")

plt.tight_layout()
plt.savefig("eda_01_sentiment_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved eda_01_sentiment_distribution.png")

# Topic category & subcategory breakdown

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cat_counts = df["topic_category"].value_counts()
axes[0].barh(cat_counts.index, cat_counts.values, color=CAT_COLORS, edgecolor="white", linewidth=0.5)
axes[0].set_title("Posts by category")
axes[0].set_xlabel("Records")
for i, v in enumerate(cat_counts.values):
    axes[0].text(v + 3, i, str(v), va="center", fontsize=9)

sub_counts = df["topic_subcategory"].value_counts().head(10)
axes[1].barh(sub_counts.index[::-1], sub_counts.values[::-1],
             color="#378ADD", edgecolor="white", linewidth=0.5)
axes[1].set_title("Top 10 subcategories")
axes[1].set_xlabel("Records")

plt.suptitle("Topic breakdown", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("eda_02_topic_breakdown.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved eda_02_topic_breakdown.png")

# Most frequent words

all_words   = " ".join(df["combined_text"]).split()
word_freq   = Counter(all_words).most_common(20)
words, freq = zip(*word_freq)

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(list(words)[::-1], list(freq)[::-1], color="#378ADD", edgecolor="white", linewidth=0.5)
ax.set_title("Top 20 most frequent words", fontsize=13, fontweight="bold")
ax.set_xlabel("Frequency")
plt.tight_layout()
plt.savefig("eda_03_word_frequency.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved eda_03_word_frequency.png")


# Word clouds per sentiment

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, sent in enumerate(["positive", "negative", "neutral"]):
    text = " ".join(df[df["sentiment"] == sent]["combined_text"])
    wc = WordCloud(width=400, height=250, background_color="white",
                   colormap="Blues" if sent == "positive" else "Oranges" if sent == "negative" else "Greys",
                   max_words=60, collocations=False).generate(text)
    axes[i].imshow(wc, interpolation="bilinear")
    axes[i].axis("off")
    axes[i].set_title(f"{sent.capitalize()} posts", fontsize=11, fontweight="bold",
                      color=COLORS[sent])

plt.suptitle("Word clouds by sentiment", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("eda_04_wordclouds.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved eda_04_wordclouds.png")

# Text length distribution

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for sent, color in COLORS.items():
    subset = df[df["sentiment"] == sent]["text_length"]
    axes[0].hist(subset, bins=20, alpha=0.6, label=sent, color=color)
axes[0].set_title("Headline length distribution by sentiment")
axes[0].set_xlabel("Character count")
axes[0].set_ylabel("Frequency")
axes[0].legend()

for sent, color in COLORS.items():
    subset = df[df["sentiment"] == sent]["word_count"]
    axes[1].hist(subset, bins=15, alpha=0.6, label=sent, color=color)
axes[1].set_title("Word count distribution by sentiment")
axes[1].set_xlabel("Word count")
axes[1].legend()

plt.suptitle("Text length analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("eda_05_text_length.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved eda_05_text_length.png")

# Engagement & trend score analysis

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Boxplot engagement by sentiment
df.boxplot(column="engagement_score", by="sentiment", ax=axes[0, 0],
           patch_artist=True, medianprops={"color": "black"})
axes[0, 0].set_title("Engagement score by sentiment")
axes[0, 0].set_xlabel("")

# Boxplot trend score by sentiment
df.boxplot(column="trend_score", by="sentiment", ax=axes[0, 1],
           patch_artist=True, medianprops={"color": "black"})
axes[0, 1].set_title("Trend score by sentiment")
axes[0, 1].set_xlabel("")

# Engagement by source
df.groupby("source")["engagement_score"].mean().plot(kind="bar", ax=axes[1, 0],
    color=CAT_COLORS[:3], edgecolor="white", linewidth=0.5)
axes[1, 0].set_title("Avg engagement by source")
axes[1, 0].set_xlabel("")
axes[1, 0].tick_params(axis="x", rotation=15)

# Trend score by category
df.groupby("topic_category")["trend_score"].mean().sort_values().plot(kind="barh",
    ax=axes[1, 1], color="#1D9E75", edgecolor="white", linewidth=0.5)
axes[1, 1].set_title("Avg trend score by category")
axes[1, 1].set_xlabel("Mean trend score")

plt.suptitle("Engagement & trend analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("eda_06_engagement.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved eda_06_engagement.png")

# Country & source breakdown

cross = pd.crosstab(df["country"], df["sentiment"])
cross.plot(kind="bar", color=[COLORS[s] for s in cross.columns],
           figsize=(8, 4), edgecolor="white", linewidth=0.5)
plt.title("Sentiment by country", fontsize=13, fontweight="bold")
plt.xlabel("")
plt.xticks(rotation=0)
plt.legend(title="Sentiment")
plt.tight_layout()
plt.savefig("eda_07_country_sentiment.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved eda_07_country_sentiment.png")

# Summary stats

print("\nDataset Summary")
print(f"Total records  : {len(df):,}")
print(f"Date range     : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Avg text length: {df['text_length'].mean():.1f} chars")
print(f"Avg word count : {df['word_count'].mean():.1f} words")
print("\nEngagement by sentiment:")
print(df.groupby("sentiment")["engagement_score"].agg(["mean","std"]).round(1))
print("\nTrend score by sentiment:")
print(df.groupby("sentiment")["trend_score"].agg(["mean","std"]).round(3))
print("\nCategory distribution:")
print(df["topic_category"].value_counts())
