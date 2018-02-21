'''
@author Oskari Järvinen
A test of how to do machine learning script. Made sure that it had outliers in data.
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Palkkatiedot.csv',sep=",")
X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 1/3, random_state= 2)

regression = LinearRegression(n_jobs=-1) # n_jobs = -1 will make your computer work hard :=)
regression.fit(X_train,y_train, sample_weight= None)

y_predictionTest = regression.predict(X_test)
x_predictionTrain = regression.predict(X_train)
# visualization

plt.scatter(X_train, y_train, color = 'black')
plt.plot(X_train,x_predictionTrain,color = 'red' )
plt.title('Palkka vs vuosia yrityksessä')
plt.xlabel('Vuosia yrityksessä')
plt.ylabel('Palkka')
plt.show()

