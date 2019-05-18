# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 22:24:12 2018

@author: hejia
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
# import data
df=pd.read_csv('C:\\Users\\hejia\\Documents\\python\\machine learning\\teleCust1000t.csv')
all_columns=df.columns
# series method value_counts()
df['gender'].value_counts()
# use hist to see distribution
plt.hist(df['income'])
for column in all_columns:
    plt.figure(figsize=(10,6))
    plt.hist(df[column],label=column)
    plt.legend(loc='best')
    
# convert Pandas data frame to Numpy array
X=df[all_columns[0:-1]].values
y=df[all_columns[-1]].values

# Normalize data
from sklearn import preprocessing
X=preprocessing.StandardScaler().fit(X).transform(X.astype(float))

# train and test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=4)

#train and predict
k=4
knn=KNeighborsClassifier(k)
neigh=knn.fit(X_train,y_train)
y_pred=neigh.predict(X_test)

# evaluation
accuracy=metrics.accuracy_score(y_test,y_pred)

# evaluate the relationship between k and accuracy
accuracy=[]
for k in range(1,15):
    knn=KNeighborsClassifier(k)
    y_pred=knn.fit(X_train,y_train).predict(X_test)
    accuracy.append(metrics.accuracy_score(y_test,y_pred))
plt.figure(figsize=(10,6))
plt.plot(range(1,15),accuracy,label='K and accuracy')
plt.xlabel('k')
plt.ylabel('accuracy_score')
plt.legend(loc='best')
    
