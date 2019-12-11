import pandas as pd
import numpy as np 
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


train = pd.read_csv("YelpData/Yelp_train.csv")
print(train.shape)
train = train.loc[~train['sentiment_score'].isnull()]
print(train.shape)
reviews = train["text"].values.tolist()
scores = train["sentiment_score"].values.tolist()

def cleanData(review):
    # remove punctuation and extraneous characters
    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    # remove excess whitespace
    review = review.strip()
    # lowercase 
    review = REPLACE_NO_SPACE.sub("", review.lower())
    review = REPLACE_WITH_SPACE.sub(" ", review)
    # remove newline characters
    review = review.replace("\n", " ")
    return review

reviews = list(map(cleanData, reviews))

print(reviews[:10])

cv = CountVectorizer(binary=True)
cv.fit(reviews)
X = cv.transform(reviews).toarray()

new_cols = cv.get_feature_names()
print(new_cols[:10])
new_train = pd.DataFrame(X)
new_train.columns = new_cols

print(new_train.shape)
print(train.shape)

dist = new_train.sum(axis = 0).values
def ifelse(condition, action1, action2):
    if (condition):
        return action1
    else:
        return action2

distlist = dist.tolist()
remove = list(map(lambda x: ifelse(x <  len(distlist)*0.0001, 1, 0), distlist))

# number of features we're keeping
print(45213 - sum(remove))

removeCols = list(filter(lambda x: x!="", list(map(lambda x: ifelse(x[0] == 1, x[1], ""), list(zip(remove, new_cols))))))
new_train.drop(removeCols, axis = 1, inplace = True)

train = pd.concat([train, new_train], axis = 1)
train.drop(["text", "name", "city", "categories", "date"], axis = 1, inplace = True)
train = train.dropna().reset_index(drop = True)

print(train.head(10))

train.to_csv("train1.csv")