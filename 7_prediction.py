# File 7 - Live Tweet Prediction
# Concept: Real time ML prediction on new unseen data

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data/Tweets.csv")
df = df[['text', 'airline_sentiment']]

# Train model
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['airline_sentiment']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

print("Model trained successfully!")
print("\n--- TWEET SENTIMENT PREDICTIONS ---\n")

# Test tweets
test_tweets = [
    "The flight was delayed and service was terrible",
    "Amazing flight experience loved the service",
    "The flight was okay nothing special",
    "Worst airline ever never flying again",
    "Thank you for the wonderful experience",
    "My luggage was lost and nobody helped me"
]

for tweet in test_tweets:
    vec = vectorizer.transform([tweet])
    prediction = model.predict(vec)
    
    if prediction[0] == 'positive':
        emoji = "😊"
    elif prediction[0] == 'negative':
        emoji = "😞"
    else:
        emoji = "😐"
    
    print(f"Tweet: {tweet}")
    print(f"Sentiment: {prediction[0]} {emoji}")
    print("---")