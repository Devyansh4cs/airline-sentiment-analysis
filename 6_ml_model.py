# File 6 - Machine Learning Model
# Concept: Text Classification using Supervised Learning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("data/Tweets.csv")
df = df[['text', 'airline_sentiment']]

# Split data into X and y
X = df['text']
y = df['airline_sentiment']

print("Total samples:", len(df))
print("Training on 80% data, Testing on 20% data")

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Convert text to numbers using TF-IDF
print("\nConverting text to numbers using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train ML model
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Test model
y_pred = model.predict(X_test_vec)

# Print results
print("\n--- MODEL RESULTS ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))