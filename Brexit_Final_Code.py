#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 20:17:21 2019

@author: racheltan
"""
import re
import pandas as pd  
import numpy as np
import nltk 
from nltk.corpus import stopwords 
nltk.download('stopwords')  
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import pickle 
from nltk.tokenize import word_tokenize
import spacy
import time
from sklearn import svm
from sklearn.model_selection import cross_validate
import csv
import matplotlib.pyplot as plt
import datetime 
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#training and test tweets taken from: 
#https://www.kaggle.com/c/twitter-sentiment-analysis2/data

train_tweets = pd.read_csv("/Users/racheltan/Desktop/train.csv", encoding='latin-1') #100k tweets

train_data = train_tweets[["Sentiment", "SentimentText"]] 
#train_data = train_data[:30000] #is this enough
train_data = (train_data.sample(n=10000))

X_data = train_data.iloc[:, 1].values #so that it converts to arrays
y_data = train_data.iloc[:, 0].values

#cleaning
processed_train = []
for tweet in range(0, len(X_data)): 
    cleaned_train = re.sub(r'http\S+', '', str(X_data[tweet])) #removes the urls in the tweets
    cleaned_train = re.sub('[^a-zA-Z]+', ' ', cleaned_train) #remove special characters and numbers
    cleaned_train = re.sub(r'\s+[a-zA-Z]\s+', ' ', cleaned_train) #removes single character
    cleaned_train = re.sub(r'\^[a-zA-Z]\s+', ' ', cleaned_train) #removes single characters from the start 
    cleaned_train = re.sub(r'\s+', ' ', cleaned_train, flags=re.I) #replaces multiple spaces with 1 space 
    cleaned_train = cleaned_train.lower() #to lowercase
    processed_train.append(cleaned_train) #now the information exists as a list 

#removing stopwords and tokenization
stop = stopwords.words('english')
no_stop_train = [] #no_stop_tweets is a nested list 
for item in processed_train: 
    no_stop_1 = [word for word in item.split() if word not in stop] 
    no_stop_train.append(' '.join(no_stop_1))
 
#lemmatize
start_time = time.time()
train_lemma = [] #not 100%, but does some of the job
nlp = spacy.load('en_core_web_sm')
for doc in nlp.pipe(no_stop_train, disable=["ner", "parser"], batch_size = 20):
    token = [token.lemma_ for token in doc]
    train_lemma.append(" ".join(token))
print("Lemmatization took", time.time() - start_time, "to run") #just to time how long it takes 

### Vectorizing the training tweets 
tfidfconverter = TfidfVectorizer()
my_vec_tfidf = TfidfVectorizer()
X = tfidfconverter.fit_transform(processed_train).toarray() #X corresponds to my_xform_tfidf in class examples 
X_col = tfidfconverter.get_feature_names()
X = pd.DataFrame(X, columns=X_col)

#Note: possible to set certain arguments for the vectorizer 
#tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
# maxfeatures - only the 2000 most frequent words 
# min_df - word must occur in at least 5 documents 
# must not occur in more than 70% of the document - too common also not as significant 

#### MODEL 1 - RANDOM FOREST with train-test-split 80-20 vs. cross validation ####
X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=0.2, random_state=0)
text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)  
text_classifier.fit(X_train, y_train)
predictions = text_classifier.predict(X_test) #75.35% accurate, but 80-20 is not the best way to split 
print(confusion_matrix(y_test,predictions))  #diagonals of a confusion matrix show the number of accurate predictions 
print(classification_report(y_test,predictions))  
print(accuracy_score(y_test, predictions)) 

#let's try cross validation instead 
cv = cross_validate(text_classifier, X, y_data, cv=5)
print(cv['test_score'])
print(cv['test_score'].mean()) #72.48% accuracy with the cross validation! 
#filename = 'basicbrexit.pkl'
#basicbrexit = pickle.dump(text_classifier, open(filename,'wb'))

#### MODEL 2 - RANDOM FOREST + PCA + GRID SEARCH ####

#determine number of components required to achieve user specified var_target 
def iterate_var(my_xform_tfidf_in, var_target, data_slice):
    var_fig = 0.0
    cnt = 1
    while var_fig <= var_target:
        pca = PCA(n_components=cnt)
        my_dim = pca.fit_transform(my_xform_tfidf_in[0:data_slice])
        var_fig = sum(pca.explained_variance_ratio_)   
        cnt += 1
    cnt -= 1
    print (cnt)
    pca = PCA(n_components=cnt)
    my_dim = pca.fit_transform(my_xform_tfidf_in)
    var_fig = sum(pca.explained_variance_ratio_) 
    print (var_fig)

    return my_dim, pca

my_dim, pca = iterate_var(X, 0.95, 100) 
#more variance explained by pca model then the better the model will be 
#91
#0.1388064554048549 Explained Variance Ratio 

def grid_search_func(param_grid, the_mode_in, the_vec_in, the_lab_in):
    grid_search = GridSearchCV(the_mode_in, param_grid=param_grid, cv=5, n_jobs = -1)
    best_model = grid_search.fit(the_vec_in, the_lab_in)
    max_score = grid_search.best_score_
    best_params = grid_search.best_params_

    return best_model, max_score, best_params
#best shows us the best accuracy that we can get  
#opt_params shows us the optimum parameters that will get us the best accuracy

param_grid = {'max_features': ['auto', 'sqrt'],
             'max_depth': [1, 2, 5, 10, 20],
             'n_estimators': [100, 200, 500],
             'random_state': [0]}
  
#call up model from above
clf_pca = RandomForestClassifier()
#call up grid search optimal
gridsearch_model, best, opt_params = grid_search_func(
        param_grid, clf_pca, X, y_data) #gives the optimal from a cross validation
#results: an accuracy of 60.95% when using max_depth 20, max_features 'auto', n_estimators '100'
#clf_pca.set_params(**gridsearch_model.best_params_) 
#clf_pca.fit(my_dim, y_data) #training the model on the best parameters 
 


#### MODEL 3 - SVM MODEL + PCA + gridsearch  #### 

## SVM model with gridsearch and PCA## 
def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds, n_jobs = -1)
    best_modelSVC = grid_search.fit(X, y)
    max_scoreSVC = grid_search.best_score_
    best_paramsSVC = grid_search.best_params_
    return best_modelSVC, max_scoreSVC, best_paramsSVC

model_SVC, best_SVC, opt_params_SVC = svc_param_selection(my_dim, y_data, 5)
#results: best is 69.76! opt_params_SVC is {C:10, gamma: 1 }

SVC_classifier = svm.SVC(kernel = 'rbf')
SVC_classifier.set_params(**model_SVC.best_params_) 
SVC_classifier.fit(my_dim, y_data)
#filename_SVC = 'SVCbrexit.pkl'
#pickle.dump(SVC_classifier, open(filename_SVC,'wb'))


###### TARGET DATA  #####

#load the target data 
total2 = pd.read_csv("/Users/racheltan/Desktop/total2.csv", sep = ';', quoting = csv.QUOTE_NONE , error_bad_lines = False)

### CLEANING THE DATA ###
#removing the columns and extra headings that I do not need 
target_data = (total2
                      .drop([0])
                      .drop(columns = ["retweets", "favorites", "geo", 
                                       "mentions", "hashtags", "id", "permalink"])
                      .drop_duplicates(subset = "text")
                      .dropna(subset = ["text"]))

#selecting the data that we need 
target_data[['Date','Time']] = target_data.date.str.split(expand=True) 
target_data = target_data[target_data.Date.str.contains("2016")] #only looking at tweets for 2016
target_data['Date'] = pd.to_datetime(target_data.Date)
target_data['Date'].dt.strftime('%Y-%m-%d')

#visualizing our results 
tweets_count = target_data.groupby('Date').count()
tweets_count['Date'] = tweets_count.index
x = tweets_count['Date']
y = tweets_count['text']
plt.scatter(x, y)
plt.show() #I used tableau to visualize this data better for the presentation  

#for now let's just look at the sentiment on the 21 of June 
data_2016 = target_data[(target_data['Date'] == datetime.date(2016,6,21))]

#make a wordcloud for hashtags
hashtags = []
for tweet in data_2016["text"]: 
    tag = re.findall(r"(#)(\s?)(\w+)", tweet)
    for element in tag:
        tag_nospace = re.sub(r'(#)(\s+)', '',''.join(element), flags=re.I)
        common_hash = ['brexit', 'Brexit', 'BREXIT']
        if tag_nospace not in common_hash: 
            hashtags.append(tag_nospace)
        
cloud_hashtag = " ".join(hashtags)
wordcloud = WordCloud(width=2500, height=2000, collocations=False).generate(cloud_hashtag)
wordcloud.to_file("Brexit_hashtags.png")
plt.figure(1,figsize=(13, 13))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Hashtags')
plt.show()   

mentions = []
for tweet in target_data["text"]: 
    at = re.findall(r"(@)(\s?)(\w+)", tweet)
    for element in at:
        at_nospace = re.sub(r'(@)(\s+)', '',''.join(element), flags=re.I)
        mentions.append(at_nospace)

cloud_mention = " ".join(mentions)
wordcloud = WordCloud(width=2500, height=2000, collocations=False).generate(cloud_mention)
#wordcloud.to_file("Brexit_mentions.png")
plt.figure(1,figsize=(13, 13))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Hashtags')
plt.show()     #a word cloud for people who were mentioned with @

#clean the tweets! 
processed_tweets = []
for tweet in data_2016["text"]: 
    cleaned_tweet = tweet.lower() #to lowercase
    cleaned_tweet = re.sub(r'http\S+', '', cleaned_tweet) #removes the urls in the tweets
    cleaned_tweet = re.sub('[^a-zA-Z]+', ' ', cleaned_tweet) #remove special characters and numbers
    cleaned_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', cleaned_tweet) #removes single character
    cleaned_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', cleaned_tweet) #removes single characters from the start ???
    cleaned_tweet = re.sub(r'\s+', ' ', cleaned_tweet, flags=re.I) #replaces multiple spaces with 1 space 
    processed_tweets.append(cleaned_tweet) #now the information exists as a list 

non_english = [' la ', ' le ', ' el ', ' que ', ' del ', ' cette ', ' cet ',
               ' es un ', ' es une ', ' dans ', ' du ', ' de ', ' ist ']
english_tweets = [tweet for tweet in processed_tweets if (any(n in tweet for n in non_english) == False)] 

stop = stopwords.words('english')
no_stop_tweets = [] #no_stop_tweets is a nested list 
for item in english_tweets: 
    no_stop = [word for word in item.split() if word not in stop] 
    no_stop_tweets.append(' '.join(no_stop))

#Tokenization if not using split function 
#tokenized_tweets = []
#for tweet in no_stop_tweets:
#    print(word_tokenize(tweet))
#    tokenized_tweets.append(word_tokenize(tweet))

#Lemmatization
start_time = time.time()
my_lemma = [] #not 100%, but does some of the job
nlp = spacy.load('en_core_web_sm')
for doc in nlp.pipe(no_stop_tweets, disable=["ner", "parser"], batch_size = 20):
    token = [token.lemma_ for token in doc]
    my_lemma.append(" ".join(token))
print("Lemmatization took", time.time() - start_time, "to run") #just to time how long it takes 


brexit_exclude = ['brexit', 'twitter']
cloud_text = " ".join(my_lemma)
cloud_text_nb = []
for word in cloud_text.split():
    if word not in brexit_exclude:
        cloud_text_nb.append(word)
final_cloud = " ".join(cloud_text_nb)
wordcloud = WordCloud(width=2500, height=2000, collocations=False).generate(final_cloud)
wordcloud.to_file("Brexit_words.png")
plt.figure(1,figsize=(13, 13))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Brexit Week')
plt.show() 

##### CLASSIFICATION #######

#Classification using Model 1 
start_time = time.time()
Brexit_X = tfidfconverter.transform(my_lemma).toarray() #X corresponds to my_xform_tfidf in class examples 
Brexit_X_col = tfidfconverter.get_feature_names()
Brexit_X = pd.DataFrame(Brexit_X, columns=Brexit_X_col)
Brexit_predict = text_classifier.predict(Brexit_X)
print("Prediction took", time.time() - start_time, "to run") #took 2500 seconds to run 
sentiment, count = np.unique(Brexit_predict, return_counts=True)
print(np.asarray((sentiment, count)).T) 
#24998 negative and 166346 positive tweets


#Classification using Model 3
#pca = PCA(n_components=91)
start_time = time.time()
Brexit_X = tfidfconverter.transform(my_lemma).toarray() #X corresponds to my_xform_tfidf in class examples 
Brexit_X_col = tfidfconverter.get_feature_names()
Brexit_X = pd.DataFrame(Brexit_X, columns=Brexit_X_col)
Brexit_X_pca = pca.transform(Brexit_X)
#pickle.dump(Brexit_X_pca, open("Brexit_X_pca",'wb'))
#SVC_model = pickle.load( open('SVCbrexit.pkl', "rb" ) )
#SVC_model.predict(Brexit_X_pca)
start_time = time.time()
prediction_pca = SVC_classifier.predict(Brexit_X_pca)
print("Prediction took", time.time() - start_time, "to run")
sentiment, count = np.unique(prediction_pca, return_counts=True)
print(np.asarray((sentiment, count)).T) 
#16245 negative 175099 positive! 










































