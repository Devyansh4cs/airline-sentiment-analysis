import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import plotly.express as px

st.title("✈️ Airline Sentiment Analysis")
st.write("Analyze customer sentiments from airline tweets")

@st.cache_data
def load_and_train():
    df = pd.read_csv("Tweets.csv")
    df = df[['text', 'airline_sentiment', 'airline']]
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['text'])
    y = df['airline_sentiment']
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    return df, model, vectorizer

df, model, vectorizer = load_and_train()

# Section 1 - Sentiment Distribution
st.subheader("📊 Overall Sentiment Distribution")
sentiment_counts = df['airline_sentiment'].value_counts()
fig1 = px.pie(
    values=sentiment_counts.values,
    names=sentiment_counts.index,
    title="Sentiment Distribution"
)
st.plotly_chart(fig1)

# Section 2 - Airline wise sentiment
st.subheader("✈️ Airline-wise Sentiment Comparison")
airline_sentiment = df.groupby(
    ['airline', 'airline_sentiment']
).size().reset_index(name='count')

fig2 = px.bar(
    airline_sentiment,
    x='airline',
    y='count',
    color='airline_sentiment',
    barmode='group',
    title="Airline-wise Sentiment"
)
st.plotly_chart(fig2)

# Section 3 - Predict your own tweet
st.subheader("🔍 Predict Your Own Tweet")
user_tweet = st.text_input("Enter any tweet here:")

if user_tweet:
    vec = vectorizer.transform([user_tweet])
    prediction = model.predict(vec)
    
    if prediction[0] == 'positive':
        st.success(f"Sentiment: {prediction[0]} 😊")
    elif prediction[0] == 'negative':
        st.error(f"Sentiment: {prediction[0]} 😞")
    else:
        st.warning(f"Sentiment: {prediction[0]} 😐")