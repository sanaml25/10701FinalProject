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


train = pd.read_csv("YelpData/Yelp_train.csv")
print(train.shape)
train = train.loc[~train['sentiment_score'].isnull()]
print(train.shape)
reviews = train["text"].values.tolist()
scores = train["sentiment_score"].values.tolist()

test = pd.read_csv("YelpData/Yelp_test.csv")
testreviews = test["text"].values.tolist()
testscores = test["sentiment_score"].values.tolist()


def cleanData(review):
    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    review = review.strip()
    review = REPLACE_NO_SPACE.sub("", review.lower())
    review = REPLACE_WITH_SPACE.sub(" ", review)
    review = review.replace("\n", " ")
    return review


reviews = list(map(cleanData, reviews))
testreviews = list(map(cleanData, testreviews))


cv = CountVectorizer(binary=True)
cv.fit(reviews)
X = cv.transform(reviews).toarray()
X_test = cv.transform(testreviews).toarray()

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


y = train["y"]
x = train.drop("y", axis = 1)
print(y.head(10))


x_train, x_val, y_train, y_val = train_test_split(x, y, train_size = 0.75)


#train.to_csv("train.csv")

print("starting")
lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', fit_intercept=True, max_iter=1000, verbose = 1).fit(x_train, y_train)
print("done!")


print ("Accuracy (Train): %s"% (accuracy_score(y_train, lr.predict(x_train))))
print ("Accuracy (Test): %s"% (accuracy_score(y_val, predict(x_val))))

