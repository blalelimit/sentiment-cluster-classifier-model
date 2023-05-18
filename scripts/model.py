from scripts.process import *


# PreProcess sentiment
def preprocess(sentiment):
    hashtags = get_hashtags()
    sentiment_cleaned = Process(sentiment)
    sentiment_cleaned.preprocess_entry(hashtags)
    sentiment_vectorized = sentiment_cleaned.vectorize_entry()
    return sentiment_vectorized


# Classify through Classifier Algorithms
def classify(sentiment, model, cluster) -> str:
    prediction = predict_entry(sentiment, model, cluster)
    return prediction
    

# Modified K-Means Clustering PCA & Percentile
def cluster(sentiment) -> str:
    prediction = predict_cluster(sentiment=sentiment)
    return prediction
