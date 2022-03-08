# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 18:31:08 2022

@author: tyler
"""

#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

#Working directory
import os 
os.chdir("C:/Users/tyler/Desktop/Python for Data Science and Machine Learning Bootcamp/20-Natural-Language-Processing")
cwd = os.getcwd()
print(cwd)

#Download stopwords
nltk.download_shell()

#Looking at messages
messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]
print(len(messages))

#Print out first 10 messages and number them
for message_no, message in enumerate(messages[:10]):
    print(message_no, message)
    print('\n')

#Reading data
messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                           names=["label", "message"])


#Exploratory data analysis
messages.head()
messages.describe()
messages.groupby('label').describe()

#Feature engineering (extracted features from text data)
#A. Length of messages
messages['length'] = messages['message'].apply(len)
messages.head()

#B. Upper case words
def freq_upper(text):
    count = sum(map(str.isupper, text.split()))
    return count

messages['upper_words'] = messages['message'].apply(freq_upper)

#C. Number of unique words
def freq_unique(text):
    unique_text = set(text.split())
    count = len(unique_text)
    return count

messages['unique'] = messages['message'].apply(freq_unique)

#Data visualization
messages['length'].plot(bins = 50, kind = 'hist')
messages['upper_words'].plot(bins = 30, kind = 'hist')
messages['unique'].plot(bins = 50, kind = 'hist')

messages.length.describe()
messages.upper_words.describe()
messages.unique.describe()

messages.hist(column= 'length', by='label', bins=50, figsize=(12,4))
messages.hist(column= 'upper_words', by='label', bins=30, figsize=(12,4))
messages.hist(column= 'unique', by='label', bins=30, figsize=(12,4))

#Text Pre-Processing
from nltk.corpus import stopwords
import string

#Function to remove stopwords and punctuation
def text_process(text):
    
    #Remove punctuation
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    #Remove stopwords
    nostop = nopunc.split()
    clean_text = [word for word in nostop if word.lower() not in stopwords.words('english')]
    
    return clean_text

#Tokenize messages
messages['message'].apply(text_process)
messages.head()

#Vectorization
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

# Print total number of vocab words
print(len(bow_transformer.vocabulary_))

#Transforming messages to bag of words object
messages_bow = bow_transformer.transform(messages['message'])
print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)

#Sparsity
sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))

#Term weighting using TD-IDF (frequency-inverse document frequency)

#This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus
#The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(messages_bow)

#Sample TD-IDF for word university
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])

#Transform entire corpus
messages_tfidf = tfidf_transformer.transform(messages_bow)

#Train-test split
from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = \
train_test_split(messages['message'], messages['label'], test_size=0.2)

#Creating pipeline
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

#Training model using bayes classifier (can use any classifying model)
pipeline.fit(msg_train, label_train)

#Predictions
predictions = pipeline.predict(msg_test)
print(predictions[0:5])

#Evaluation
from sklearn.metrics import classification_report
print (classification_report(label_test, predictions))















































