from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf, col, lower, regexp_replace
from pyspark.sql.types import ArrayType, StringType

# ─────────────────────────────────────────────
# Load the Dataset
# ─────────────────────────────────────────────
df = spark.table("default.social_analysis_refined")
print("Dataset loaded:", df.count())
df.select("headline").show(5, truncate=50)

# ─────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────
df = df.withColumn("cleaned_headline", lower(col("headline")))
df = df.withColumn("cleaned_headline", regexp_replace(col("cleaned_headline"), r"http\S+|www\S+", ""))
df = df.withColumn("cleaned_headline", regexp_replace(col("cleaned_headline"), r"[^a-z\s]", ""))
df = df.withColumn("cleaned_headline", regexp_replace(col("cleaned_headline"), r"\s+", " "))

df.select("headline", "cleaned_headline").show(3, truncate=60)

# ─────────────────────────────────────────────
# Tokenization and Stopword Removal
# ─────────────────────────────────────────────
tokenizer = RegexTokenizer(inputCol="cleaned_headline", outputCol="tokens", pattern="\\s+")
remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")

# Run tokenization and stopword removal first
pipeline_prep = Pipeline(stages=[tokenizer, remover])
df = pipeline_prep.fit(df).transform(df)

# ─────────────────────────────────────────────
# Lemmatization via UDF
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

lemmatize_udf = udf(lambda tokens: [simple_lemmatize(t) for t in tokens], ArrayType(StringType()))
df = df.withColumn("lemmatized_tokens", lemmatize_udf(col("filtered_tokens")))

df.select("cleaned_headline", "filtered_tokens", "lemmatized_tokens").show(3, truncate=50)

# ─────────────────────────────────────────────
# TF-IDF
# ─────────────────────────────────────────────
hashingTF = HashingTF(inputCol="lemmatized_tokens", outputCol="raw_features", numFeatures=100)
idf = IDF(inputCol="raw_features", outputCol="tfidf_features")

pipeline_tfidf = Pipeline(stages=[hashingTF, idf])
result = pipeline_tfidf.fit(df).transform(df)

result.select("headline", "lemmatized_tokens", "tfidf_features").show(3, truncate=50)

# ─────────────────────────────────────────────
# Save Outputs
# ─────────────────────────────────────────────
cols = ["id", "date", "source", "language", "country",
        "topic_category", "topic_subcategory",
        "headline", "cleaned_headline","processed_headline",
        "sentiment", "engagement_score", "trend_score"]

result = result.withColumn("processed_headline", 
    array_join(col("lemmatized_tokens"), " "))

result.select(cols).write.mode("overwrite").saveAsTable("default.preprocessed_data")
result.select("id", "sentiment", "tfidf_features").write.mode("overwrite").saveAsTable("default.tfidf_features")

print("Saved to default.preprocessed_data and default.tfidf_features")

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
print("Total records:", result.count())
print("TF-IDF features: 100")
result.groupBy("sentiment").count().show()
