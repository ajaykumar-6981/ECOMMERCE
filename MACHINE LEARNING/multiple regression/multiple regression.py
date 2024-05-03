
# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# importing the dataset
dataset=pd.read_csv('50_Startups.csv')

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

print(x)
print(y)

# encoding the categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])] ,remainder="passthrough")
X=np.array(ct.fit_transform(x))

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

print(X_train)
print(X_test)
print(y_train)
print(y_test)


# applying the multiple regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


# predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))# visualizing the data
# plt.scatter(X_train,y_train)
# plt.plot(X_train,y_train)
# plt.show()
