# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 20:04:58 2018

@author: hejia
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
# import data
df=pd.read_csv('drug200.csv')
columns=df.columns
X=df[columns[:-1]].values
y=df[columns[-1]]

# transfer text values to numerical
sex_code=preprocessing.LabelEncoder()
sex_code.fit(['F','M'])
X[:,1]=sex_code.transform(X[:,1])

BP_code=preprocessing.LabelEncoder()
BP_code.fit(['LOW','NORMAL','HIGH'])
X[:,2]=BP_code.transform(X[:,2])

chol_code=preprocessing.LabelEncoder()
chol_code.fit(['NORMAL','HIGH'])
X[:,3]=chol_code.transform(X[:,3])

X_trainset, X_testset, y_trainset, y_testset=train_test_split(X,y,test_size=0.3,random_state=3)


# model with decision tree
drugTree=DecisionTreeClassifier(criterion='entropy',max_depth=4)
drugTree.fit(X_trainset,y_trainset)
predTree=drugTree.predict(X_testset)

# evaluation
print('decision tree accuracy: ',metrics.accuracy_score(y_testset,predTree))









