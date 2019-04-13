# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 10:44:57 2018

@author: hejia
"""
#   data
# 1. dataset
from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data
y=iris.target

# 2. data split: train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=4)

#-----------modules------------
# 1. Classifier
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier (n_neighbors=5)
knn.fit(X,y)
y_pred=knn.predict([[3,5,4,2]])

# 2. LogisticRegression
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X,y)
y_pred=logreg.predict([[3,5,4,2]])

# 3. linear regression
from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(X_train,y_train)
linreg.intercept_
linreg.coef_
y_pred=linreg.predict(X_test)

# -----------evaluation-----------------
# accuracy_score works for logisticRegression and Classifier
from sklearn import metrics
metrics.accuracy_score(y,y_pred) # true, pred

# MAE or MSE works for linear regression
metrics.mean_absolute_error(y_test,y_pred)

# ---------Corss validation-----------------
from sklearn.model_selection import KFold
kf=KFold(n_splits=5).split(range(25))
from sklearn.model_selection import cross_val_score
knn=KNeighborsClassifier(n_neighbors=5)
scores=cross_val_score(knn,X,y,cv=10,scoring='accuracy')
