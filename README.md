# ✈️ Airline Sentiment Analysis

## 📌 Project Overview
A complete NLP and Machine Learning project that analyzes 
customer sentiments from airline tweets. 
Built an interactive web application that predicts 
sentiment of any tweet in real time.

## 🎯 Business Problem
Airlines need to understand customer feedback at scale.
This project analyzes 14,640 tweets to extract insights
about customer satisfaction across major US airlines.

## 📊 Key Findings
- United Airlines had the most negative tweets (2633)
- Southwest Airlines had the most positive tweets (570)
- Overall 62.7% tweets were negative across all airlines

## 🛠️ Technologies Used
- Python
- Pandas
- Matplotlib
- WordCloud
- Scikit-learn
- Plotly
- Streamlit
- NLTK

## 📁 Project Structure

    social media sentiment analysis/
    │
    ├── data/
    │   └── Tweets.csv
    │
    ├── 1_sentiment_analysis.py
    ├── 2_bar_pie_chart.py
    ├── 3_wordcloud.py
    ├── 4_sentiment_wordcloud.py
    ├── 5_airline_comparison.py
    ├── 6_ml_model.py
    ├── 7_prediction.py
    ├── app.py
    └── README.md

## ⚙️ How To Run

### Install dependencies
    pip install pandas matplotlib wordcloud nltk scikit-learn plotly streamlit

### Run individual analysis files
    python 1_sentiment_analysis.py
    python 2_bar_pie_chart.py
    python 3_wordcloud.py
    python 4_sentiment_wordcloud.py
    python 5_airline_comparison.py
    python 6_ml_model.py
    python 7_prediction.py

### Run Web Application
    streamlit run app.py

## 🤖 ML Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | 80% |
| Negative F1 | 0.88 |
| Positive F1 | 0.70 |
| Neutral F1 | 0.59 |

## 📈 Project Pipeline
Raw Tweets → Data Cleaning → Sentiment Analysis → Visualization → ML Model → Web Application

## 💡 Key Concepts Learned
- NLP Text Preprocessing
- Sentiment Analysis
- TF-IDF Vectorization
- Logistic Regression
- Data Visualization
- Streamlit Web Apps
- Class Imbalance Problem