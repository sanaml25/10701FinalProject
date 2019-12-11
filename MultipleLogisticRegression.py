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
import pickle

output = open("train3.pkl", "rb")
train = pickle.load(output)
train = train.dropna()

print(train.head(10))

y = train["y"]
x = train.drop("y", axis = 1)
print(y.head(10))

x_train, x_val, y_train, y_val = train_test_split(x, y, train_size = 0.75)

#train.to_csv("train.csv")

print("starting")
lr = linear_model.LogisticRegression(multi_class='multinomial', solver='sag', fit_intercept=True, max_iter=100, verbose = 1).fit(x_train, y_train)
print("done!")

from sklearn.metrics import confusion_matrix

print ("Accuracy (Train): %s"% (accuracy_score(y_train, lr.predict(x_train))))

confusionMatrix = confusion_matrix(y_train, lr.predict(x_train))
print(confusionMatrix)

print ("Accuracy (Test): %s"% (accuracy_score(y_val, lr.predict(x_val))))

confusionMatrix = confusion_matrix(y_val, lr.predict(x_val))
print(confusionMatrix)

output = open("logisticRegression4.pkl", "wb")
pickle.dump(lr, output)

