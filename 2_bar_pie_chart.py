# File 2 - Bar Chart and Pie Chart
# Concept: Data Visualization using Matplotlib

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/Tweets.csv")
df = df[['text', 'airline_sentiment']]

# Count sentiments
sentiment_counts = df['airline_sentiment'].value_counts()

# BAR CHART
plt.figure(figsize=(8, 5))
sentiment_counts.plot(kind='bar', color=['red', 'blue', 'green'])
plt.title("Sentiment Distribution - Bar Chart")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# PIE CHART
plt.pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct='%1.1f%%',
    colors=['red', 'blue', 'green']
)
plt.title("Sentiment Distribution - Pie Chart")
plt.show()