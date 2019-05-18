# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 10:20:10 2018

@author: hejia
"""
# import modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model

data=pd.read_csv('Fuel.csv')
data.head()

cdata=data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

# generate training and testing data: 80% for train
msk=np.random.rand(len(data))<0.8
train=cdata[msk]
test=cdata[~msk]

# use linear regression model
regr=linear_model.LinearRegression()
x=np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y=np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x,y)
print('the coefficient is: {}'.format(regr.coef_))

plt.figure(figsize=(12,7))
y_pred=regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
test_x=np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
test_y=np.asanyarray(test[['CO2EMISSIONS']])
plt.plot(y_pred)
plt.plot(test_y,color='r')
plt.xlabel('different factors')
plt.ylabel('CO2 emission')

# evaluate the model
print('mean of error square is {}'.format(np.mean((y_pred-test_y)**2)))
print('score is {}'.format(regr.score(test_x,test_y)))


