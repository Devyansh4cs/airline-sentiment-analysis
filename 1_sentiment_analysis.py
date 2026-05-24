# File 1 - Basic Sentiment Analysis
# Concept: Loading data and understanding distribution

import pandas as pd

# Load dataset
df = pd.read_csv("data/Tweets.csv")

# Keep important columns
df = df[['text', 'airline_sentiment']]

# Count sentiments
sentiment_counts = df['airline_sentiment'].value_counts()

# Print results
print("Total Tweets:", len(df))
print("\nSentiment Distribution:")
print(sentiment_counts)

# Business Insight
print("\nBusiness Insight:")
print("Most common sentiment:", sentiment_counts.index[0])
print("Least common sentiment:", sentiment_counts.index[-1])