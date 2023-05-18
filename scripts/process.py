from pandas import read_csv, DataFrame
from joblib import load
from sklearn.preprocessing import normalize
from scripts.preprocess import *


# INITIALIZATION METHODS

# Get hashtags list
def get_hashtags():
    hashtags = read_csv('datasets/hashtags_list.csv', sep=',', header=None)
    return flatten(hashtags.values)


# Input entry/entries of strings
def input_sentiment():
    sentiment = input('Input your sentiment: ')
    return DataFrame([sentiment], columns={'text'})


# Input entry/entries of strings
def input_model() -> str:
    model = input('Input the model (CNB/SVM/LR/RF/ENS): ')
    return model


# CLUSTERING METHODS

def predict_cluster(sentiment) -> str:
    kmeans = load('models/modkmeans_pp.pkl')
    pca = load('models/pca.pkl')
    sentiment_pca = pca.transform(sentiment)
    cluster_prediction = kmeans.predict(sentiment_pca)[0]
    return cluster_prediction


# CLASSIFICATION METHODS

# Load classifier model and predict
def predict_entry(sentiment, model_name, cluster) -> str:
    if model_name not in ['cnb', 'svm', 'lr', 'rf', 'ens']:
        model_name = 'ens'
    # if cluster not in [str(x) for x in range(6)]:
    #     cluster = '0'

    model = load(f'models/{model_name}_cluster{cluster}.pkl')
    prediction = model.predict(sentiment).tolist()[0]
    result = f'{map_model(model_name)} in Cluster {cluster} predicted {map_prediction(prediction)}.'

    return result


# Load classifier model and predict on all clusters
def predict_all(sentiment, model_name):
    if model_name not in ['cnb', 'svm', 'lr', 'rf', 'ens']:
        model_name = 'ens'
        print('ENS running by default')

    for x in range(6):
        model = load(f'models/{model_name}_cluster{str(x)}.pkl')
        print(f'{map_model(model_name)} on Cluster {str(x)} predicted {map_prediction(model.predict(sentiment)[0])}.')


def map_model(model) -> str:
    dict_map = {'cnb': 'COMPLEMENT NAIVE BAYES', 'svm': 'SUPPORT VECTOR MACHINE', 
                'lr': 'LOGISTIC REGRESSION', 'rf': 'RANDOM FOREST', 'ens': 'COMBINATION ENSEMBLE MODEL'}
    return dict_map.get(model)


def map_prediction(prediction) -> str:
    dict_map = {-1: 'NEGATIVE', 0: 'NEUTRAL', 1: 'POSITIVE'}
    return dict_map.get(prediction)


# PREPROCESSING METHODS

class Process:
    def __init__(self, sentiment):
        self.sentiment = sentiment


    # Preprocess entry
    def preprocess_entry(self, hashtags):
        self.sentiment = PreProcess(self.sentiment, hashtags)
        self.sentiment.lower()
        self.sentiment.cleaning_a()
        self.sentiment.cleaning_b()
        self.sentiment.tokenization()
        return self.sentiment.lemmatization()


    # Tfidf Vectorizer
    def vectorize_entry(self):
        df = self.sentiment.get_sentiment()
        if df['text_preprocessed'][0] == '[]':
            return list()
        else:
            tfidf = load('models/tfidf.pickle')
            x = tfidf.fit_transform(df['text_preprocessed'])
            df = DataFrame(normalize(x).toarray(), columns=tfidf.get_feature_names_out())
            if (df.T[0] == 0.0).all():
                return list()
            else:
                return df
