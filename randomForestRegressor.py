import pandas as pd 
import numpy as np 
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import mord
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn import metrics
import pickle 

output = open("train4.pkl", "rb")
train = pickle.load(output)
train = train.dropna()

print(train.head(10))


y = train["y"]
x = train.drop("y", axis = 1)
print(y.head(10))


x_train, x_val, y_train, y_val = train_test_split(x, y, train_size = 0.75)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 100 decision trees
rf = RandomForestRegressor(n_estimators = 100, verbose = 1)
# Train the model on training data
rf.fit(x_train, y_train)

# Use the forest's predict method on the test data
y_pred = rf.predict(x_val)
# Calculate the absolute errors
errors = abs(y_pred - y_val)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_val, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_val, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))

y_pred = rf.predict(x_train)
# Calculate the absolute errors
errors = abs(y_pred - y_train)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))

output = open("randomForestRegressor4.pkl", "wb")
pickle.dump(rf, output)
