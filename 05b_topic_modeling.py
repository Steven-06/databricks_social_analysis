import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})
CAT_COLORS = ["#1D9E75", "#378ADD", "#D85A30", "#BA7517", "#888780"]

# Load 
df = pd.read_csv("preprocessed_data.csv")
df["date"] = pd.to_datetime(df["date"])
print(f"Loaded {len(df):,} records")

# Build document-term matrix 
cv = CountVectorizer(
    max_features=400,
    min_df=3,          # ignore very rare words
    max_df=0.95,       # ignore very common words 
    ngram_range=(1, 2),
    stop_words="english",
)
X_counts = cv.fit_transform(df["processed_headline"].fillna(""))
vocab = cv.get_feature_names_out()
print(f"Vocabulary size: {len(vocab):,} | Doc-term matrix: {X_counts.shape}")

# Fit LDA with 5 topics

N_TOPICS  = 5
TOPIC_NAMES = [      
    "Sports & markets",
    "Global events",
    "AI & technology",
    "Economy & jobs",
    "Entertainment",
]

lda = LatentDirichletAllocation(
    n_components=N_TOPICS,
    max_iter=30,
    learning_method="online",
    learning_offset=50.0,
    random_state=42,
    n_jobs=-1,
)
lda.fit(X_counts)
print("LDA perplexity:", round(lda.perplexity(X_counts), 2))

# Topic distribution per document
topic_probs   = lda.transform(X_counts)                   # (n_docs, n_topics)
df["lda_topic"]      = topic_probs.argmax(axis=1)
df["lda_topic_name"] = df["lda_topic"].map(dict(enumerate(TOPIC_NAMES)))
df["lda_conf"]       = topic_probs.max(axis=1)

# Top words per topic
top_n = 12
topics = []
for idx, comp in enumerate(lda.components_):
    top_words = [vocab[i] for i in comp.argsort()[:-top_n - 1:-1]]
    top_probs = np.sort(comp)[::-1][:top_n]
    topics.append({
        "id": idx,
        "name": TOPIC_NAMES[idx],
        "words": top_words,
        "probs": top_probs / top_probs.sum(),   # normalize for display
    })

# text summary

with open("lda_topics.txt", "w") as f:
    f.write("="*60 + "\n")
    f.write("LDA TOPIC MODELING — RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Number of topics : {N_TOPICS}\n")
    f.write(f"Perplexity       : {lda.perplexity(X_counts):.2f}  (lower = better)\n")
    f.write(f"Vocabulary size  : {len(vocab):,}\n\n")
    for t in topics:
        f.write(f"{'─'*40}\n")
        f.write(f"Topic {t['id']+1}: {t['name']}\n")
        f.write("  Top words: " + ", ".join(t["words"]) + "\n")
    f.write(f"\n{'─'*40}\n")
    f.write("Topic distribution per document:\n")
    f.write(df["lda_topic_name"].value_counts().to_string())

print("Saved lda_topics.txt")

# Plot 1: Top words per topic 

fig, axes = plt.subplots(1, N_TOPICS, figsize=(18, 5))
for i, (t, ax) in enumerate(zip(topics, axes)):
    ax.barh(t["words"][::-1], t["probs"][::-1],
            color=CAT_COLORS[i], edgecolor="white", linewidth=0.5)
    ax.set_title(f"Topic {i+1}\n{t['name']}", fontsize=10, fontweight="bold",
                 color=CAT_COLORS[i])
    ax.set_xlabel("Weight")
    ax.tick_params(axis="y", labelsize=9)
    if i > 0:
        ax.set_yticklabels([])

plt.suptitle("LDA — Top words per topic", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("lda_01_top_words.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved lda_01_top_words.png")

# Plot 2: Topic distribution across documents

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

topic_counts = df["lda_topic_name"].value_counts()
axes[0].bar(range(len(topic_counts)), topic_counts.values,
            color=CAT_COLORS, edgecolor="white", linewidth=0.5)
axes[0].set_xticks(range(len(topic_counts)))
axes[0].set_xticklabels(topic_counts.index, rotation=15, ha="right", fontsize=9)
axes[0].set_title("Documents per topic")
axes[0].set_ylabel("Count")

# Mean topic probability across corpus
mean_probs = topic_probs.mean(axis=0)
axes[1].bar(range(N_TOPICS), mean_probs,
            color=CAT_COLORS, edgecolor="white", linewidth=0.5)
axes[1].set_xticks(range(N_TOPICS))
axes[1].set_xticklabels([f"T{i+1}" for i in range(N_TOPICS)])
axes[1].set_title("Mean topic probability in corpus")
axes[1].set_ylabel("Avg. probability")

plt.suptitle("LDA — Topic distribution", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("lda_02_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved lda_02_distribution.png")

# Plot 3: Document topic confidence distribution

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["lda_conf"], bins=30, color="#378ADD", edgecolor="white", linewidth=0.5)
ax.axvline(df["lda_conf"].mean(), color="#D85A30", linestyle="--", linewidth=1.5,
           label=f"Mean: {df['lda_conf'].mean():.2f}")
ax.set_xlabel("Max topic probability (confidence)")
ax.set_ylabel("Count")
ax.set_title("Topic assignment confidence", fontsize=12, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("lda_05_confidence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved lda_05_confidence.png")

print("\nTopic modeling complete.")
print(f"  Perplexity: {lda.perplexity(X_counts):.2f}")
print("\nTopic sizes:")
print(df["lda_topic_name"].value_counts())
