# 📊 Social Media Trending Topics Analysis 2026

A comprehensive data science project analyzing synthetic social media trends for 2026. This repository contains a complete pipeline for data processing, exploratory analysis, sentiment analysis, topic modeling, and trend analysis using both local Python analysis and Databricks integration.

## 🎯 Project Overview

This project analyzes trending social media topics across multiple dimensions:
- **Exploratory Data Analysis (EDA)** - Statistical insights and visualizations
- **Sentiment Analysis** - Multi-class sentiment classification using machine learning models
- **Topic Modeling** - Latent Dirichlet Allocation (LDA) for topic discovery
- **Trend Analysis** - Temporal trends and evolution of topics
- **Model Evaluation** - Comprehensive performance metrics and cross-validation

## 📁 Project Structure

```
databricks_social_analysis/
├── 03_EDA.py                      # Exploratory Data Analysis with visualizations
├── 05a_sentiment_model.py         # Sentiment classification (Positive/Neutral/Negative)
├── 05b_topic_modeling.py          # LDA-based topic modeling
├── 06_trend_analysis.py           # Temporal trend analysis and forecasting
├── 07_evaluation.py               # Model evaluation and cross-validation
├── formatter.py                   # Data cleaning and preprocessing (Spark)
├── NLP/
│   └── NLP.py                     # PySpark NLP pipeline for text processing
├── data/
│   └── trending_topics_2026_synthetic.csv    # Synthetic dataset
└── README.md
```

## 🚀 Getting Started

### Prerequisites

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn wordcloud scipy pyspark
```

### 1. Data Processing

Start with data cleaning using `formatter.py` (Databricks/PySpark):

```python
# Loads CSV data, applies type enforcement, and stores as Delta Table
df = spark.read.csv("/path/to/trending_topics_2026_synthetic.csv", header='True', inferSchema='True')
df_final = df \
    .withColumn("date", to_date(col("date"))) \
    .withColumn("engagement_score", col("engagement_score").cast(IntegerType())) \
    .withColumn("trend_score", col("trend_score").cast(DoubleType()))
df_final.write.mode("overwrite").saveAsTable("default.social_analysis_refined")
```

### 2. Run Analysis Pipeline

Execute scripts in order:

```bash
python 03_EDA.py                  # Explore data distributions and patterns
python 05a_sentiment_model.py     # Train sentiment classifier
python 05b_topic_modeling.py      # Discover topics using LDA
python 06_trend_analysis.py       # Analyze temporal trends
python 07_evaluation.py           # Evaluate all models
```

## 📊 Analysis Modules

### 03_EDA.py - Exploratory Data Analysis
- Statistical summaries and distributions
- Sentiment breakdowns
- Engagement and trend score analysis
- Word clouds and frequency analysis

### 05a_sentiment_model.py - Sentiment Analysis
- **Models:** Logistic Regression, Random Forest, Gradient Boosting, Linear SVC
- **Features:** TF-IDF vectorization with preprocessing
- **Output:** Classification reports, confusion matrices, ROC curves

### 05b_topic_modeling.py - Topic Modeling
- Latent Dirichlet Allocation (LDA) with 400-word vocabulary
- Document-term matrix construction
- Topic distribution analysis

### 06_trend_analysis.py - Trend Analysis
- Time-series visualization by category
- Sentiment trends over time
- Word clouds by topic

### 07_evaluation.py - Model Evaluation
- Cross-validation (StratifiedKFold)
- Comprehensive metrics (Accuracy, F1, Precision, Recall)
- Model comparison and performance analysis

### NLP/NLP.py - PySpark NLP Pipeline
- Tokenization, stop word removal, and hashing TF-IDF
- Text cleaning with regex patterns
- Integration with Databricks/Delta Lake

## 🔧 Configuration

### Color Palette (Consistent Across Analysis)
- **Sentiment:** Blue (positive), Gray (neutral), Orange (negative)
- **Categories:** Green, Blue, Orange, Brown, Gray

### Data Schema
- `date`: DateType
- `headline`: StringType
- `sentiment`: StringType (Positive/Neutral/Negative)
- `engagement_score`: IntegerType
- `trend_score`: DoubleType
- `category`: StringType

## 📈 Key Features

✅ End-to-end data pipeline with Databricks Delta Lake integration  
✅ Multiple ML models for sentiment classification  
✅ Advanced NLP techniques (TF-IDF, LDA, tokenization)  
✅ Comprehensive visualization and analysis outputs  
✅ Cross-validation and rigorous model evaluation  
✅ Time-series trend analysis  

## 📝 Notes

- The project uses synthetic data for demonstration
- All scripts expect preprocessed data in CSV format
- Databricks notebooks (formatter.py, NLP/NLP.py) require PySpark environment
- Local analysis scripts use pandas, scikit-learn, and matplotlib

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