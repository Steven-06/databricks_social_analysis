import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# ─────────────────────────────────────────────
# Stopwords
# ─────────────────────────────────────────────
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

# ─────────────────────────────────────────────
# Load the Cleaned Dataset
# ─────────────────────────────────────────────
df = spark.table("default.social_analysis_refined").toPandas()
print("Dataset loaded:", df.shape)
print(df['headline'].head())

# ─────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)         # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()      # fix whitespace
    tokens = [w for w in text.split() if w not in STOPWORDS]
    return ' '.join(tokens)

df['cleaned_headline'] = df['headline'].apply(clean_text)
print(df[['headline', 'cleaned_headline']].head(3))

# ─────────────────────────────────────────────
# Tokenization and Lemmatization
# ─────────────────────────────────────────────
def simple_lemmatize(word):
    if word.endswith('ing') and len(word) > 5:
        return word[:-3]
    if word.endswith('tion') and len(word) > 6:
        return word[:-4]
    if word.endswith('ments') and len(word) > 7:
        return word[:-5]
    if word.endswith('ment') and len(word) > 6:
        return word[:-4]
    if word.endswith('ers') and len(word) > 5:
        return word[:-3]
    if word.endswith('er') and len(word) > 4:
        return word[:-2]
    if word.endswith('es') and len(word) > 4:
        return word[:-2]
    if word.endswith('s') and len(word) > 3:
        return word[:-1]
    return word

def tokenize_and_lemmatize(text):
    tokens = re.findall(r'\b[a-z]+\b', text)
    return [simple_lemmatize(t) for t in tokens]

df['tokens'] = df['cleaned_headline'].apply(tokenize_and_lemmatize)
df['processed_headline'] = df['tokens'].apply(lambda t: ' '.join(t))
print(df[['cleaned_headline', 'tokens', 'processed_headline']].head(3))

# ─────────────────────────────────────────────
# TF-IDF Feature Engineering
# ─────────────────────────────────────────────
vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2), min_df=2)
tfidf_matrix = vectorizer.fit_transform(df['processed_headline'])
feature_names = vectorizer.get_feature_names_out()

print("TF-IDF shape:", tfidf_matrix.shape)
print("Sample features:", list(feature_names[:10]))

# ─────────────────────────────────────────────
# Save the TF-IDF Features
# ─────────────────────────────────────────────

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
tfidf_df.insert(0, 'id', df['id'].values)
tfidf_df.insert(1, 'sentiment', df['sentiment'].values)
tfidf_df.to_csv("tfidf_features.csv", index=False)

cols = ['id', 'date', 'source', 'language', 'country',
        'topic_category', 'topic_subcategory',
        'headline', 'cleaned_headline', 'processed_headline',
        'sentiment', 'engagement_score', 'trend_score']
df[cols].to_csv("preprocessed_data.csv", index=False)

print("Saved tfidf_features.csv and preprocessed_data.csv")

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────

print(f"Total records: {len(df)}")
print(f"TF-IDF features: {len(feature_names)}")
print(df['sentiment'].value_counts())