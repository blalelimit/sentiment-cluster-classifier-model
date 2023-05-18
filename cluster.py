from scripts.model import preprocess
from scripts.process import *


# Clustering Process
def cluster_all(sentiment):
    sentiment_vectorized = preprocess(sentiment=sentiment)

    if len(sentiment_vectorized) != 0:
        cluster_prediction = predict_cluster(sentiment=sentiment_vectorized)
        print(f'MODIFIED K-MEANS CLUSTERING (PCA & PERCENTILE) predicted Cluster {cluster_prediction}.')
    else:
        print('Invalid input or your sentiment contains unrecognized text')


if __name__ == '__main__':
    sentiment = input_sentiment()
    cluster_all(sentiment=sentiment)
