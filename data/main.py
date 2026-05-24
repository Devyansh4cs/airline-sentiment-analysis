import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("Tweets.csv")

# Keep important columns
df = df[['text', 'airline_sentiment']]

# Split data
X = df['text']
y = df['airline_sentiment']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text to numbers
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train ML model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Test model
y_pred = model.predict(X_test_vec)

# Print accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# Predict your own tweets
print("\n--- YOUR OWN TWEET PREDICTIONS ---")

tweet1 = ["The flight was delayed and service was terrible"]
tweet2 = ["Amazing flight experience loved the service"]
tweet3 = ["The flight was okay nothing special"]

for tweet in [tweet1, tweet2, tweet3]:
    vec = vectorizer.transform(tweet)
    prediction = model.predict(vec)
    print(f"Tweet: {tweet[0]}")
    print(f"Predicted Sentiment: {prediction[0]}")
    print("---")