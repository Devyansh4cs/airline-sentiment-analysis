# File 3 - Word Cloud
# Concept: Text Mining and NLP Visualization

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Load dataset
df = pd.read_csv("data/Tweets.csv")
df = df[['text', 'airline_sentiment']]

# BASIC WORD CLOUD
print("Showing Basic Word Cloud...")
all_words = ' '.join(df['text'])

wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white'
).generate(all_words)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud)
plt.axis('off')
plt.title("Basic Word Cloud")
plt.show()

# CLEANED WORD CLOUD
print("Showing Cleaned Word Cloud...")
stop_words = set(stopwords.words('english'))

filtered_words = []
for tweet in df['text']:
    words = tweet.split()
    for word in words:
        if word.lower() not in stop_words:
            filtered_words.append(word)

clean_text = ' '.join(filtered_words)

wordcloud_clean = WordCloud(
    width=1000,
    height=500,
    background_color='white'
).generate(clean_text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_clean)
plt.axis('off')
plt.title("Cleaned Word Cloud - Stopwords Removed")
plt.show()