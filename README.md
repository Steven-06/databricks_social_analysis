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

The `cleaner.ipynb` notebook performs the following "Power Moves" to ensure data quality:

* **Deduplication:** Ensures every `id` is unique.
* **Type Casting:** Converts `date` strings into actual Date objects and scores into proper numeric types.
* **Standardization:** Normalizes `sentiment` and `language` casing to prevent duplicate categories in charts.

## 🛠️ Tech Stack

* **Storage:** Unity Catalog (Volumes & Tables)
* **Compute:** Databricks Serverless (Spark Connect)
* **Language:** PySpark / SQL
* **Version Control:** GitHub Integration