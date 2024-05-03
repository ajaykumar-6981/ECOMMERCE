
#importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing the dataset
dataset=pd.read_csv('Position_Salaries.csv')

x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

print(y)
print(x)

from sklearn.linear_model import LinearRegression
le=LinearRegression()
le.fit(x,y)

# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)

plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

