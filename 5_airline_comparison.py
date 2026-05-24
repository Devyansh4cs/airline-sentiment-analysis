# File 5 - Airline Wise Sentiment Comparison
# Concept: Comparative Analysis across categories

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/Tweets.csv")
df = df[['text', 'airline_sentiment', 'airline']]

# Group by airline and sentiment
airline_sentiment = df.groupby(
    ['airline', 'airline_sentiment']
).size().unstack()

# Print insights
print("Airline wise Sentiment Counts:")
print(airline_sentiment)
print("\nAirline with most negative tweets:")
print(airline_sentiment['negative'].idxmax())
print("\nAirline with most positive tweets:")
print(airline_sentiment['positive'].idxmax())

# Plot
airline_sentiment.plot(
    kind='bar',
    figsize=(12, 6),
    color=['blue', 'orange', 'green']
)

plt.title("Airline-wise Sentiment Comparison")
plt.xlabel("Airline")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=45)
plt.legend(title="Sentiment")
plt.tight_layout()
plt.show()