from pyspark.sql.functions import col, to_date
from pyspark.sql.types import IntegerType, DoubleType

# 1. Load the data using the standard method you provided
# Using inferSchema='True' here allows Spark to take that initial pass to detect types
df = spark.read.csv("/Volumes/workspace/default/social_data/trending_topics_2026_synthetic.csv", header='True', inferSchema='True')

# 2. Type Enforcement & Cleaning
# Refining the schema to ensure consistent data types for analysis
df_final = df \
    .withColumn("date", to_date(col("date"))) \
    .withColumn("engagement_score", col("engagement_score").cast(IntegerType())) \
    .withColumn("trend_score", col("trend_score").cast(DoubleType()))

# 3. Save as a Managed Delta Table
# Overwriting the existing table to keep the 'refined' version as the source of truth
df_final.write.mode("overwrite").saveAsTable("default.social_analysis_refined")

# 4. Verify the result
# Using display() is perfect for Databricks environments to see the tabular results
display(spark.table("default.social_analysis_refined"))