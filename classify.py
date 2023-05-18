from scripts.model import preprocess
from scripts.process import *


# Classification Process
def classify_all(sentiment, model_name):
    sentiment_vectorized = preprocess(sentiment=sentiment)

    if len(sentiment_vectorized) != 0:
        predict_all(sentiment_vectorized, model_name)
    else:
        print('Invalid input or your sentiment contains unrecognized text')


if __name__ == '__main__':
    sentiment = input_sentiment()
    model = input_model().lower()
    classify_all(sentiment=sentiment, model_name=model)
