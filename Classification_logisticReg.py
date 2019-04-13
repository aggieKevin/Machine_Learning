# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 05:09:50 2018

@author: hejia
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# import data
df=pd.read_csv('ChurnData.csv')

columns=['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']
df=df[columns]
df['churn']=df['churn'].astype('int') # convert the data type
X=df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']].values
y=df['churn'].values

#preprocess the data
X=preprocessing.StandardScaler().fit(X).transform(X)
X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=4)

logreg=LogisticRegression().fit(X_train,y_train)
y_pred=logreg.predict(X_test)

# evaluation

