
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# importing the dataset

dataset=pd.read_csv('Ecommerce_data.csv')

x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:, -1].values


# print(x)
# print(y)
from sklearn.preprocessing import OneHotEncoder

# Assuming 'sku_data' is a list/array of SKUs



from sklearn.ensemble import RandomForestRegressor

regressor=RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(x,y)

regressor.predict([[6.5]])
# print(regressor.predict([6.5]))

x_grid=np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape(len(x_grid), 1)
plt.scatter(x,y,color='red')

plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title("Truth or Bluff")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
