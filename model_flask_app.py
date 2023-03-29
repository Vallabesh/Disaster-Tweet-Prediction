import numpy as np 
import pandas as pd
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import re
from sklearn.externals import joblib

from bs4 import BeautifulSoup
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import math
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input,Dense,LSTM
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Input , Dropout, Flatten
import h5py
from tensorflow.python.keras.callbacks import TensorBoard,ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
from time import time
import pickle 
from numpy import zeros
from tensorflow.keras.layers import (
    BatchNormalization)
from keras.layers import Bidirectional
import keras
from flask import Flask, jsonify, request

import flask
app = Flask(__name__)

def decontracted(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"\'I'm", "I am", phrase)
    phrase = re.sub(r"I\'ve", "I have", phrase)
    phrase = re.sub(r"isn\'t", "is not", phrase)
    phrase = re.sub(r"we\'ll", "we will", phrase)
    phrase = re.sub(r"we\'re", "we are", phrase)
    phrase = re.sub(r"we\'ve", "we have", phrase)
    phrase = re.sub(r"weren\'t", "were not", phrase)
    phrase = re.sub(r"quake", "earthquake", phrase)
    phrase = re.sub(r"nado", "tornado", phrase)
    return phrase

def clean_tweet(sentence):
    stopwords = set(STOPWORDS)
    sentence = re.sub(r"http\S+", "", sentence)
    sentence = BeautifulSoup(sentence, 'lxml').get_text()
    sentence = decontracted(sentence)
    sentence = re.sub("\S*\d\S*", "", sentence).strip()
    sentence = re.sub('[^A-Za-z]+', ' ', sentence)
    sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in stopwords)
    return sentence.strip()

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index_flask')
def index():
    return flask.render_template('index_flask.html')


@app.route('/predict_tweet', methods=['POST'])

def predict_tweet():
    model_flask = keras.models.load_model('model_flask')
    tokenizer_tweet = joblib.load('token_tweet.pkl')
    to_predict_list = request.form.to_dict()
    
    tweet = clean_tweet(to_predict_list['tweet'])
    tweet=  [tweet]
    tweet_token_text  = tokenizer_tweet.texts_to_sequences(tweet)
    tweet_token_text = pad_sequences(tweet_token_text, maxlen=35 , padding='post')
    pred = (model_flask.predict(tweet_token_text) > 0.5).astype("int32")
    if pred == 0:
        return "It's a Hoax"
    else:
        return "Real disaster!!"
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)    