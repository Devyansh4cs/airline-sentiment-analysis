import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Airline Sentiment Analysis",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    .metric-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .positive {color: #00cc96;}
    .negative {color: #ef553b;}
    .neutral {color: #636efa;}
    </style>
""", unsafe_allow_html=True)

# Load and train model
@st.cache_data
def load_and_train():
    df = pd.read_csv("data/Tweets.csv")
    df = df[['text', 'airline_sentiment', 'airline']]

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['text'])
    y = df['airline_sentiment']

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return df, model, vectorizer

df, model, vectorizer = load_and_train()

# Sidebar
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/3125/3125713.png",
    width=100
)
st.sidebar.title("✈️ Airline Sentiment")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Dashboard", "🔍 Predict Tweet", "📂 Bulk Analysis", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Total Tweets Analyzed:**")
st.sidebar.markdown(f"### {len(df):,}")

# ─── HOME PAGE ───
if page == "🏠 Home":
    st.title("✈️ Airline Sentiment Analysis")
    st.markdown("### Welcome to the Airline Sentiment Analytics Dashboard")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Tweets", f"{len(df):,}")
    with col2:
        negative = len(df[df['airline_sentiment'] == 'negative'])
        st.metric("Negative Tweets", f"{negative:,}", "-62.7%")
    with col3:
        positive = len(df[df['airline_sentiment'] == 'positive'])
        st.metric("Positive Tweets", f"{positive:,}", "+16.1%")
    with col4:
        neutral = len(df[df['airline_sentiment'] == 'neutral'])
        st.metric("Neutral Tweets", f"{neutral:,}", "+21.2%")

    st.markdown("---")
    st.markdown("## 🎯 Project Overview")
    st.info("""
    This project analyzes **14,640 tweets** from major US airlines
    to understand customer sentiment and extract business insights.
    
    **Key Finding:** United Airlines had the most negative tweets
    while Southwest Airlines had the best customer satisfaction.
    """)

    st.markdown("## 🛠️ Technologies Used")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("Python | Pandas | NLTK")
    with col2:
        st.info("Scikit-learn | TF-IDF")
    with col3:
        st.warning("Streamlit | Plotly")

# ─── DASHBOARD PAGE ───
elif page == "📊 Dashboard":
    st.title("📊 Analytics Dashboard")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['airline_sentiment'].value_counts()
        fig1 = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Airline wise Sentiment")
        airline_sentiment = df.groupby(
            ['airline', 'airline_sentiment']
        ).size().reset_index(name='count')
        fig2 = px.bar(
            airline_sentiment,
            x='airline',
            y='count',
            color='airline_sentiment',
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.subheader("🏆 Key Business Insights")
    col1, col2, col3 = st.columns(3)

    with col1:
        most_negative = df[df['airline_sentiment'] == 'negative']['airline'].value_counts().index[0]
        st.error(f"😞 Most Negative Airline\n\n**{most_negative}**")

    with col2:
        most_positive = df[df['airline_sentiment'] == 'positive']['airline'].value_counts().index[0]
        st.success(f"😊 Most Positive Airline\n\n**{most_positive}**")

    with col3:
        most_tweets = df['airline'].value_counts().index[0]
        st.info(f"📢 Most Tweeted Airline\n\n**{most_tweets}**")

    st.markdown("---")

    st.subheader("🔎 Analyze Specific Airline")
    selected_airline = st.selectbox(
        "Select Airline:",
        df['airline'].unique()
    )

    airline_df = df[df['airline'] == selected_airline]
    airline_counts = airline_df['airline_sentiment'].value_counts()

    col1, col2 = st.columns(2)
    with col1:
        fig3 = px.pie(
            values=airline_counts.values,
            names=airline_counts.index,
            title=f"{selected_airline} Sentiment",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.markdown(f"### {selected_airline} Stats")
        st.metric("Total Tweets", len(airline_df))
        st.metric("Negative", airline_counts.get('negative', 0))
        st.metric("Positive", airline_counts.get('positive', 0))
        st.metric("Neutral", airline_counts.get('neutral', 0))

    # Word Cloud
    st.markdown("---")
    st.subheader(f"☁️ Most Common Words - {selected_airline}")
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

    airline_words = ' '.join(airline_df['text'])
    filtered_words = ' '.join([
        w for w in airline_words.split()
        if w.lower() not in stop_words
    ])

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='RdYlGn'
    ).generate(filtered_words)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud)
    ax.axis('off')
    st.pyplot(fig)

# ─── PREDICT PAGE ───
elif page == "🔍 Predict Tweet":
    st.title("🔍 Tweet Sentiment Predictor")
    st.markdown("---")

    user_tweet = st.text_area(
        "Enter your tweet here:",
        height=100,
        placeholder="Type any airline related tweet..."
    )

    if st.button("🔍 Analyze Sentiment"):
        if user_tweet:
            vec = vectorizer.transform([user_tweet])
            prediction = model.predict(vec)
            confidence = model.predict_proba(vec).max() * 100

            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                if prediction[0] == 'positive':
                    st.success(f"## 😊 Positive")
                elif prediction[0] == 'negative':
                    st.error(f"## 😞 Negative")
                else:
                    st.warning(f"## 😐 Neutral")

            with col2:
                st.metric("Confidence Score", f"{confidence:.1f}%")

        else:
            st.warning("Please enter a tweet first!")

# ─── BULK ANALYSIS PAGE ───
elif page == "📂 Bulk Analysis":
    st.title("📂 Bulk Tweet Analysis")
    st.markdown("---")

    st.info("""
    Upload a CSV file with a column named **'text'**
    containing tweets to analyze all at once!
    """)

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv']
    )

    if uploaded_file is not None:
        upload_df = pd.read_csv(uploaded_file)

        st.success(f"✅ File uploaded! Found {len(upload_df)} rows")

        if 'text' in upload_df.columns:

            st.subheader("🔍 Analyzing tweets...")

            texts = upload_df['text'].astype(str)
            vecs = vectorizer.transform(texts)
            predictions = model.predict(vecs)
            confidences = model.predict_proba(vecs).max(axis=1) * 100

            upload_df['Predicted Sentiment'] = predictions
            upload_df['Confidence %'] = confidences.round(1)

            st.markdown("---")

            col1, col2, col3 = st.columns(3)
            pred_counts = pd.Series(predictions).value_counts()

            with col1:
                st.error(f"😞 Negative\n\n**{pred_counts.get('negative', 0)}**")
            with col2:
                st.success(f"😊 Positive\n\n**{pred_counts.get('positive', 0)}**")
            with col3:
                st.warning(f"😐 Neutral\n\n**{pred_counts.get('neutral', 0)}**")

            st.markdown("---")

            fig = px.pie(
                values=pred_counts.values,
                names=pred_counts.index,
                title="Sentiment Distribution of Uploaded Tweets",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig)

            st.subheader("📋 Detailed Results")
            st.dataframe(
                upload_df[['text', 'Predicted Sentiment', 'Confidence %']]
            )

            st.subheader("📥 Download Results")
            csv = upload_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="sentiment_results.csv",
                mime="text/csv"
            )

        else:
            st.error("❌ CSV file must have a column named 'text'!")
            st.info("Please make sure your CSV has a 'text' column!")

# ─── ABOUT PAGE ───
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("---")

    st.markdown("""
    ## 📌 Project Summary
    This is an NLP based Data Analytics project that analyzes
    customer sentiments from airline tweets.

    ## 🎯 Business Problem
    Airlines need to understand customer feedback at scale.
    This dashboard provides instant insights from 14,640 tweets.

    ## 🤖 ML Model Performance
    | Metric | Score |
    |--------|-------|
    | Accuracy | 80% |
    | Negative F1 | 0.88 |
    | Positive F1 | 0.70 |
    | Neutral F1 | 0.59 |

    ## 💡 Key Concepts Used
    - Natural Language Processing (NLP)
    - TF-IDF Vectorization
    - Logistic Regression
    - Data Visualization
    - Streamlit Dashboard
    - Class Imbalance Problem

    ## 👨‍💻 Developer
    **Devyansh**
    - GitHub: github.com/Devyansh4cs
    """)