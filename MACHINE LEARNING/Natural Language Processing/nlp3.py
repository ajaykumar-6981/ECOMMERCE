
# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing the dataset
dataset=pd.read_csv('',delimiter='\t',quoting=3)

# cleaning the dataset
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

corpus=[]
for i in range(0,270):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][1])
    review.lower()
    review.split()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review=' '.join(review)
    corpus.append(review)

print(corpus)
# creating a bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
print(len(X[0]))
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

y=dataset.iloc[:,:-1].values

y=le.fit_transform(y)
# Splitting the dataset in train set and the test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

print(X_train)
print(X_test)

# importing the navie model
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()

classifier.fit(X_train,y_train)

# predicting the test set
y_pred=classifier.predict(X_test)
print(y_pred)

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
ac=accuracy_score(y_test,y_pred)
print(ac)



