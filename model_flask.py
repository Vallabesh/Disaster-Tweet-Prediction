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
    sentence = re.sub(r"http\S+", "", sentence)
    sentence = BeautifulSoup(sentence, 'lxml').get_text()
    sentence = decontracted(sentence)
    sentence = re.sub("\S*\d\S*", "", sentence).strip()
    sentence = re.sub('[^A-Za-z]+', ' ', sentence)
    sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in stopwords)
    return sentence.strip()

df_train = pd.read_csv("train.csv")
stopwords = set(STOPWORDS)

preprocessed_tweets = []
for sentence in df_train['text'].values:
    preprocessed_tweets.append(clean_tweet(sentence))
    
Y = df_train['target'].values

tokenizer_tweet = Tokenizer()
tokenizer_tweet.fit_on_texts(preprocessed_tweets)
joblib.dump(tokenizer_tweet, 'token_tweet.pkl') 

tweet_tokenlen = len(tokenizer_tweet.word_index)+1
tweet_token_text  = tokenizer_tweet.texts_to_sequences(preprocessed_tweets)
X_pad = pad_sequences(tweet_token_text, maxlen=35 , padding='post' )
X_train, X_test, y_train, y_test = train_test_split(X_pad, Y, stratify=df_train['target'], test_size=0.20)
embedding_matrix = zeros((tweet_tokenlen, 100))


glove_model = {}
f = open('glove.6B.100d.txt',encoding="utf8")
for line in f:
    split_line  = line.split()
    word = split_line[0]
    embedding  = np.asarray(split_line [1:], dtype='float32')
    glove_model [word] = embedding 
f.close()
    
glove_keys = glove_model.keys()
for word, i in tokenizer_tweet.word_index.items():
    if word in glove_keys:
        embedding_vector = glove_model.get(word)
        embedding_matrix[i] = embedding_vector
        
filepath="weightsflask1.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy',save_best_only=True, mode='max', verbose=1)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/model_flask".format(time()))
callbacks_1 = [checkpoint,tensorboard_callback]

embedding_vecor_length = 32
model7 = Sequential()
model7.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights = [embedding_matrix], input_length=35))
model7.add(LSTM(108,return_sequences=True))
model7.add(Dropout(0.5))
model7.add(LSTM(64,return_sequences=True))
model7.add(Dropout(0.5))
model7.add(LSTM(32))
model7.add(Dropout(0.3))
model7.add(Dense(1, activation='relu'))
print(model7.summary())

model7.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
history_7 = model7.fit(X_train,
          y_train,
          batch_size=64,
          validation_data=(X_test, y_test),
          epochs=15,callbacks=[callbacks_1])

model7.save('model_flask')
        
def predict_tweet(string):
    model_flask = keras.models.load_model('model_flask')
    tokenizer_tweet = joblib.load('token_tweet.pkl')
    tweet = clean_tweet(string)
    tweet=  [tweet]
    tweet_token_text  = tokenizer_tweet.texts_to_sequences(tweet)
    tweet_token_text = pad_sequences(tweet_token_text, maxlen=35 , padding='post')
    #pred = model_flask.predict(tweet_token_text)
    pred = (model_flask.predict(tweet_token_text) > 0.5).astype("int32")
    if pred == 0:
        return "It's a Hoax"
    else:
        return "Real disaster!!"