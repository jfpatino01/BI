import re
import sys
import os
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import tokenizer

#current = os.path.dirname(os.path.realpath(__file__))
#parent = os.path.dirname(os.path.dirname(current))
#sys.path.append(parent)

class Model:

    def __init__(self):
        self.model = None
        nltk.download('stopwords')
        stop_words = set(stopwords.words('spanish'))

    def make_predictions(self, data):
        model = joblib.load('/Users/pipe/Desktop/GitHub/BI/Etapa 2/assets/modelo.joblib')
        dataP = data['review_es']
        pred = model.predict(dataP)
        return pred

    def make_predictions_one(self, data):
        model = joblib.load('/Users/pipe/Desktop/GitHub/BI/Etapa 2/assets/modelo.joblib')
        pred = model.predict([data])
        print(pred)
        return pred[0]

