from pyspark.sql.functions import col, to_date
from pyspark.sql.types import IntegerType, DoubleType

# 1. Load the data from your Volume
file_path = "/Volumes/workspace/default/social_data/trending_topics_2026_synthetic.csv"
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(file_path)

# 2. Power Move: Explicit Type Enforcement
# This ensures that even if a future CSV is messy, your "Cleaned" table remains stable
df_final = df \
    .withColumn("date", to_date(col("date"))) \
    .withColumn("engagement_score", col("engagement_score").cast(IntegerType())) \
    .withColumn("trend_score", col("trend_score").cast(DoubleType()))

# 3. Power Move: Save as a Managed Delta Table
# This "registers" the data in the Catalog so your team doesn't need file paths anymore
df_final.write.mode("overwrite").saveAsTable("default.social_analysis_refined")

# 4. Verify it's there
display(spark.table("default.social_analysis_refined"))