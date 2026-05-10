# 📊 Social Media Trending Topics Analysis 2026

Welcome to the backend infrastructure for our trend analysis project. This repository contains the data pipeline to clean, standardize, and share synthetic social media insights.

## 🏗️ Project Architecture

We use a **Medallion Architecture** light approach:

1. **Raw Data:** Located in the Unity Catalog Volume at `/Volumes/workspace/default/social_data/`.
2. **Processed Data:** Cleaned via PySpark and registered as a global Delta Table (`default.social_analysis_refined`).

## 🚀 Getting Started for Team Members

### 1. Setup your Environment

* **Link GitHub:** Go to **User Settings** > **Linked Accounts** and connect your GitHub profile.
* **Clone Repo:** In Databricks, go to **Workspace** > **Git Folders** > **Create Git folder** and paste this repo's URL.
* **Compute:** Ensure you are connected to **Serverless** compute or a shared cluster.

### 2. Accessing the Data

You do **not** need to read CSV files. The data is already live in the Catalog.

**Using Python:**

```python
df = spark.table("default.social_analysis_refined")
display(df)

```

**Using SQL:**

```sql
SELECT * FROM default.social_analysis_refined WHERE sentiment = 'positive'

```

## 🧹 Data Cleaning Logic

The `formatter.py` script ensures data quality through a direct Spark transformation pipeline. The logic is optimized for the Databricks environment to prepare raw CSV data for downstream analysis:

* **Standard Ingestion:** Utilizes `spark.read.csv` with `header='True'` and `inferSchema='True'`. This allows Spark to perform an initial scan of the synthetic social data and automatically detect basic data structures.
* **Explicit Type Enforcement:** To ensure the schema is production-ready, the script applies manual casting to key columns:
* **Date Conversion:** Transforms the `date` column into a proper `DateType` object using `to_date()`.
* **Metric Accuracy:** Casts `engagement_score` to **IntegerType** and `trend_score` to **DoubleType** to prevent precision loss during aggregation.


* **Delta Storage:** The cleaned DataFrame is persisted as a **Managed Delta Table** (`default.social_analysis_refined`). Using `mode("overwrite")` ensures that the table serves as a fresh, reliable source of truth every time the script is executed.
* **Validation:** Final verification is performed using `display()`, providing an immediate tabular view of the refined dataset within the workspace.

---

### Implementation Reference

```python
df_final = df \
    .withColumn("date", to_date(col("date"))) \
    .withColumn("engagement_score", col("engagement_score").cast(IntegerType())) \
    .withColumn("trend_score", col("trend_score").cast(DoubleType()))

```

## 🛠️ Tech Stack

* **Storage:** Unity Catalog (Volumes & Tables)
* **Compute:** Databricks Serverless (Spark Connect)
* **Language:** PySpark / SQL
* **Version Control:** GitHub Integration