
# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing the dataset
dataset=pd.read_csv('rug_sale.csv')

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

print(x)
print(y)

# splitting the dataset into train set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

print(X_train)
print(y_test)

# Predicting the Test set results
y_pred=regressor.predict(X_test)

# Visualisation the training set results

plt.scatter(X_test,y_test,color="red")
plt.plot(X_train,regressor.predict(X_train))
plt.xlabel('Year on experience V/s Sallery')
plt.show()
