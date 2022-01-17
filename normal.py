import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from pymongo import MongoClient
from matplotlib import pyplot as plt
from sqlalchemy import create_engine
import pymysql
import seaborn as sns
import datetime


pd
# CNN
# from keras.layers.convolutional import Conv1D

# RNN
# from keras.layers.recurrent import LSTM




client = MongoClient("localhost:27017")

db=client.indv

# // SELECT * 
# // FROM zipcodes
result=db.labeled.find({})
#print(result)
source=list(result)
df=pd.DataFrame(source)
df.head()


maxlen = 500
max_words=200
n_epochs = 6
n_df_size = 30000
model_name_for_sql = 'selfnn'
tosql = True


df = df[:n_df_size]


def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)


X = []
sentences = list(df['Review'])
for sen in sentences:
    X.append(preprocess_text(sen))


y = df['Label']

y = np.array(list(map(lambda x: 1 if x=="Positive" else 0, y)))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
vocab_size = len(tokenizer.word_index) + 1
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


g = df.groupby('Label')
g = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
df = g

sns.countplot(x='Label', data=df)


model = Sequential()

# input layer
model.add(Embedding(vocab_size, 256, input_length=maxlen))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())


history=model.fit(X_train, y_train, batch_size=128, epochs=n_epochs, verbose=1, validation_split=0.2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)
print("Test Score:", scores[0])
print("Test Accuracy:", scores[1])



