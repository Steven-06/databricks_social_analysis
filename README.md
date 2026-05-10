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

- **Python 3.8+**
- **pip** package manager

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn wordcloud scipy pyspark
```

Or install from requirements file (if available):
```bash
pip install -r requirements.txt
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

### 2. Prepare Input Data

The scripts expect a `preprocessed_data.csv` file in the project root directory with the following columns:

| Column | Type | Example |
|--------|------|---------|
| `date` | String (YYYY-MM-DD) | 2026-01-15 |
| `headline` | String | "New AI breakthrough announced" |
| `sentiment` | String | Positive/Neutral/Negative |
| `engagement_score` | Integer | 1500 |
| `trend_score` | Float | 8.75 |
| `category` | String | ai_and_tech |

The sample data is available in `data/trending_topics_2026_synthetic.csv`.

### 3. Run Analysis Pipeline

Execute scripts in order:

```bash
python 03_EDA.py                  # Explore data distributions and patterns
python 05a_sentiment_model.py     # Train sentiment classifier
python 05b_topic_modeling.py      # Discover topics using LDA
python 06_trend_analysis.py       # Analyze temporal trends
python 07_evaluation.py           # Evaluate all models
```

**Expected Runtime:** ~2-5 minutes total for all scripts on a standard machine

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

## � Expected Outputs

Running the analysis pipeline generates:

- **03_EDA.py:** Statistical summaries, distribution plots, word clouds
- **05a_sentiment_model.py:** Trained model files, confusion matrices, ROC curves
- **05b_topic_modeling.py:** LDA model, topic-word distributions, coherence scores
- **06_trend_analysis.py:** Time-series plots, sentiment evolution charts, trend visualizations
- **07_evaluation.py:** Cross-validation results, performance metrics, model comparisons

*Note: Output files are typically saved in the working directory or displayed in notebooks.*

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'X'`
**Solution:** Ensure all packages are installed: `pip install -r requirements.txt`

### Issue: `FileNotFoundError: preprocessed_data.csv`
**Solution:** The CSV file must exist in the project root directory. Use the sample data:
```bash
cp data/trending_topics_2026_synthetic.csv preprocessed_data.csv
```

### Issue: Scripts are very slow
**Solution:** 
- Ensure you have at least 4GB RAM available
- Close other applications to free up memory
- Use Python 3.8+ for better performance

### Issue: Matplotlib/Seaborn plots not displaying
**Solution:** Ensure you have a display environment set up. On headless systems, add to scripts:
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

### Issue: Databricks/PySpark errors in formatter.py or NLP/NLP.py
**Solution:** These scripts require a Databricks environment with Spark. For local testing, skip these files.

## 📝 Notes

- The project uses synthetic data for demonstration purposes
- All local analysis scripts expect preprocessed data in CSV format
- Databricks notebooks (formatter.py, NLP/NLP.py) require PySpark environment and Unity Catalog access
- Local analysis scripts use pandas, scikit-learn, and matplotlib
- Visualization outputs require a working display/backend environment