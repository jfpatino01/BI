import re
import sys
import os
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

#current = os.path.dirname(os.path.realpath(__file__))
#parent = os.path.dirname(os.path.dirname(current))
#sys.path.append(parent)

class Model:

    def __init__(self):
        self.model = None

    def make_predictions(self, data):
        nltk.download('stopwords')
        stop_words = set(stopwords.words('spanish'))
        
        model = joblib.load('/assets/modelo.joblib')
        pred = model.predict(data)
        return pred

# Función para tokenizar los tweets
def tokenizer(text):
    tt = TweetTokenizer()
    return tt.tokenize(text) 