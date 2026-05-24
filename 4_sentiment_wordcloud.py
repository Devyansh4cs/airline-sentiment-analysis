# File 4 - Sentiment Wise Word Cloud
# Concept: Comparative Text Analysis across sentiments

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset
df = pd.read_csv("data/Tweets.csv")
df = df[['text', 'airline_sentiment']]

# Create 3 side by side word clouds
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sentiments = ['negative', 'neutral', 'positive']

for i, sentiment in enumerate(sentiments):
    # Filter by sentiment
    subset = df[df['airline_sentiment'] == sentiment]
    
    # Get words
    words = ' '.join(subset['text'])
    
    # Remove stopwords
    filtered = ' '.join([
        w for w in words.split() 
        if w.lower() not in stop_words
    ])
    
    # Create word cloud
    wc = WordCloud(
        width=600,
        height=400,
        background_color='white'
    ).generate(filtered)
    
    # Show
    axes[i].imshow(wc)
    axes[i].axis('off')
    axes[i].set_title(
        f"{sentiment.upper()} Tweets", 
        fontsize=14
    )

plt.suptitle("Sentiment-wise Word Cloud", fontsize=16)
plt.tight_layout()
plt.show()