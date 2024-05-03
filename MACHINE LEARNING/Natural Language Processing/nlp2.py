
# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing the dataset
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t', quoting=3)


# cleaning the dataset
import nltk
import re

nltk.download('stopwords')
from nltk.corpus import stopwords
# eng=stopwords.words('english')

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

corpus = []
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
stop_words.discard('not')  # Removing 'not' from stopwords as you mentioned

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])  # Replacing non-alphabetic characters with space
    review = review.lower()  # Converting to lowercase
    review = review.split()  # Splitting into words
    review = [ps.stem(word) for word in review if word not in stop_words]  # Stemming and removing stopwords
    review = ' '.join(review)  # Joining words back into a sentence
    corpus.append(review)

# print(corpus)

# creating bag of words

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

cv=CountVectorizer()
X=cv.fit_transform(corpus).toarray()
le=LabelEncoder()
y=dataset.iloc[:,-1].values
y=le.fit_transform(y)

print(len(X[0]))

print(X)

print(y)

# Splitting the dataset into the train set and the test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

print(X_train)
print(y_train)

# applying the Naive Bay model on train set and the test set
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)

# applying prediction method
y_pred=classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
print(y_pred)

# applying confusion metrix
from sklearn.metrics import confusion_matrix,accuracy_score
cx=confusion_matrix(y_test,y_pred)
print(cx)
print(accuracy_score(y_test,y_pred))




